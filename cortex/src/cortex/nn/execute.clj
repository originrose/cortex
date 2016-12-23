(ns cortex.nn.execute
  "Executing the graph means training or inference.  The goal is to allow both imperative/effectful implementations
and pure functional implementations but to abstract common details of training or execution into
one place written in such a way that someone can affect the behavior of various implementations and design
new execution strategies (like parameter sharing) at least partially without needing to work withing a specific
implementation.  It is important to realize that training the network means essentially a transformation from
layer-graph -> layer-graph via some training process."
  (:require [cortex.nn.traverse :as traverse]
            [cortex.nn.layers :as layers]
            [cortex.dataset :as ds]
            [think.resource.core :as resource]
            [cortex.loss :as loss]
            [cortex.nn.protocols :as cp]))


(defn- safe-inc
  [num-or-nil]
  (if (nil? num-or-nil)
    1
    (inc num-or-nil)))


(defn train-seq
  "Infinite sequence of networks, one for each epoch.
The context is expected to already be bound to the network."
  [context {:keys [batch-size] :as built-network} dataset]
  (let [dataset-streams (->> (map :dataset-stream (traverse/get-dataset-bindings built-network))
                             (remove nil?)
                             set)
        dataset-epoch (ds/get-batches dataset batch-size :training dataset-streams)
        trained-network (-> (cp/train-batch-sequence context built-network dataset-epoch {})
                            last
                            (update :epoch-count safe-inc))]
    (cons trained-network
          (lazy-seq (train-seq context trained-network dataset)))))


(defn inferences->node-id-loss-pairs
  "Given the set of inferences from an inference run of the network
and the set of labels along with the bindings (traverse/get-dataset-bindings built-network)
return a map of node-id -> loss.  Note that inferences are map of node-id->batches
while labels is a map of dataset-stream->data."
  [inferences labels dataset-bindings]
  (let [inference-columns (ds/batches->columns inferences)
        label-columns (ds/batches->columns labels)
        output-nodes (->> dataset-bindings
                          (filter #(and (= :output (get % :direction))
                                        (get % :dataset-stream)
                                        (get % :loss-function))))
        node-id->output-streams (->> output-nodes
                                     (map (fn [{:keys [node-id dataset-stream]}]
                                            [node-id dataset-stream]))
                                     (into {}))
        ;;inferences are organized by node id
        ;;labels are organized by dataset stream
        inference-label-pairs (->> (keys inferences)
                                   (map (fn [node-id]
                                          [node-id [(get inference-columns node-id)
                                                    (get label-columns (get node-id->output-streams node-id))]]))
                                   (into {}))]
    (->> output-nodes
         (map (fn [{:keys [node-id loss-function]}]
                (let [[inferences labels] (get inference-label-pairs node-id)]
                  [node-id (loss/average-loss loss-function inferences labels)]))))))


(defn train-infer-seq
  "train and infer against the trained network.  This is useful for doing things like
calculating per-epoch loss.  For this to work correctly the dataset needs to return the exact
same data per batch type.
Returns map of:
{:network trained-network
 :inferences inferences from this run
 :label-fn function to call to get labels
 :dataset-bindings io bindings from the dataset to this network."
  [context {:keys [batch-size] :as built-network} dataset & {:keys [infer-batch-type]
                                                             :or {infer-batch-type :cross-validation}}]
  (let [bindings (traverse/get-dataset-bindings built-network)
        input-streams (->> bindings
                           (filter #(= :input (get % :direction)))
                           (map :dataset-stream)
                           (remove nil?)
                           set)
        output-nodes (->> bindings
                          (filter #(and (= :output (get % :direction))
                                        (get % :dataset-stream)
                                        (get % :loss-function))))
        output-streams (->> (map :dataset-stream output-nodes)
                            set)
        label-fn #(ds/get-batches dataset batch-size infer-batch-type output-streams)]
   (->> (train-seq context built-network dataset)
        (map (fn [trained-network]
               {:network trained-network
                :inferences (cp/infer-batch-sequence context trained-network
                                                     (ds/get-batches dataset batch-size
                                                                     infer-batch-type
                                                                     input-streams)
                                                     {})
                :label-fn label-fn
                :dataset-bindings bindings})))))


(defn train
  "Train a network.  Returns the last network.  Note that this does not reset the
  network's epoch count.  Trains with: (take-while pred-fn (train-infer-seq context dataset)),
  thus trains whiel pred-fn returns true given the output of train-infer-seq.
  The network epoch count is on the network (:epoch-count).

  There is an assumtion that the network has had inputs and outputs bound to it already.

  This function is meant more as an example rather than the rule of how to train a network.
  It is a good idea, for example for the predicate function to have the side effect of saving
  the network if the current loss is lower than any previous loss."
  [context built-network dataset input-bindings output-bindings pred-fn
   & {:keys [batch-size infer-batch-type optimiser disable-infer?]
      :or {batch-size 128 infer-batch-type :cross-validation
           optimiser (layers/adam)}}]
  (let [built-network (traverse/network->training-traversal
                       (assoc built-network :batch-size batch-size)
                       input-bindings output-bindings
                       :optimiser optimiser)
        train-fn (if disable-infer?
                   #(->> (train-seq context % dataset)
                         (map (fn [net] {:network net})))
                   #(train-infer-seq context % dataset :infer-batch-type infer-batch-type))]
    ;;the resource context here is very important because there are execution contexts
    ;;that allocate major resources that have to be released when they are no longer needed.
    ;;THUS we do this with the with-resource-context call.
    (resource/with-resource-context
      (as-> (cp/bind-to-network context built-network {}) trained-network
        (train-fn trained-network)
        (take-while pred-fn trained-network)
        (last trained-network)
        (get trained-network :network)
        (cp/save-to-network context trained-network {})))))


(defn infer
  "Given a network and a dataset infer a set of data.  data is returned as a map of:
node-id->data-stream."
  [context built-network dataset input-bindings output-bindings
   & {:keys [batch-size infer-batch-type]
      :or {batch-size 128 infer-batch-type :holdout}}]
  (let [built-network (traverse/network->inference-traversal built-network input-bindings output-bindings)
        built-network (assoc built-network
                             :batch-size batch-size)
        input-streams (->> (traverse/get-dataset-bindings built-network)
                           (filter #(= :input (get % :direction)))
                           (map :dataset-stream)
                           (remove nil?)
                           set)
        epoch (ds/get-batches dataset batch-size infer-batch-type input-streams)]
    (clojure.pprint/pprint epoch)
    (resource/with-resource-context
      (as-> (cp/bind-to-network context built-network {}) network-or-seq
        (cp/infer-batch-sequence context network-or-seq epoch {})
        ;;Because the resource context will close and batches->columns is lazy
        ;;we have to force the realization of the entire sequence right here else we
        ;;risk leaking deallocated resources out that will be accessed when the list
        ;;is later realized
        (vec network-or-seq)
        (ds/batches->columns network-or-seq)))))
