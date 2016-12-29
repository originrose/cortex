(ns cortex.nn.execute
  "Executing the graph means training or inference.  The goal is to allow both imperative/effectful implementations
and pure functional implementations but to abstract common details of training or execution into
one place written in such a way that someone can affect the behavior of various implementations and design
new execution strategies (like parameter sharing) at least partially without needing to work withing a specific
implementation.  It is important to realize that training the network means essentially a transformation from
layer-graph -> layer-graph via some training process.
Both train and infer should be wrapped in resource contexts; this is not done at this level.
Furthermore infer should be both wrapped in a resource context and completely realized."
  (:require [cortex.nn.traverse :as traverse]
            [cortex.nn.layers :as layers]
            [cortex.dataset :as ds]
            [think.resource.core :as resource]
            [cortex.loss :as loss]
            [cortex.nn.protocols :as cp]
            [cortex.optimise :as optimise]))


(defn- safe-inc
  [num-or-nil]
  (if (nil? num-or-nil)
    1
    (inc num-or-nil)))


(defn bind-to-network
  "Bind a context to a network.  This allows the context to setup specialized datastructures.
The network should already have a traversal at this point (traversal/network->X where X is either
training or inference).
See cortex.nn.protocols/PExecutionContext"
  [context network options]
  (cp/bind-to-network context network options))


(defn save-to-network
  "Save to the network.  If options includes save-gradients then all io buffers and gradients
are saved.  Buffers are saved into the traversal section."
  [context network options]
  ;;Save to network and remove the traversal if there is no reason to keep it.
  ;;This avoids doing something like persisting the bindings and ending up in
  ;;an odd spot.
  (cond-> (-> (cp/save-to-network context network options)
              traverse/clear-io-bindings)
    (not (contains? options :save-gradients?))
    (dissoc :traversal)))


(defn- train-seq
  "Infinite sequence of networks, one for each epoch.
The context is expected to already be bound to the network."
  [context {:keys [batch-size] :as built-network} dataset]
  (let [streams (->> (map :stream (traverse/get-io-bindings built-network))
                             (remove nil?)
                             set)
        dataset-epoch (ds/get-batches dataset batch-size :training streams)
        trained-network (-> (cp/train-batch-sequence context built-network dataset-epoch {})
                            last
                            (update :epoch-count safe-inc))]
    (cons {:network trained-network}
          (lazy-seq (train-seq context trained-network dataset)))))


(defn- train-infer-seq
  "train and infer against the trained network.  This is useful for doing things like
calculating per-epoch loss.  For this to work correctly the dataset needs to return the exact
same data per batch type.
Returns map of:
{:network trained-network
 :inferences inferences from this run
 :label-fn function to call to get labels
 :dataset-bindings io bindings from the dataset to this network."
  [context network dataset & {:keys [infer-batch-type]
                              :or {infer-batch-type
                                   :cross-validation}}]
  (let [batch-size (long (get network :batch-size))
        input-streams (traverse/get-input-streams network)]
   (->> (train-seq context network dataset)
        (map (fn [{:keys [network] :as entry}]
               (assoc entry
                      :inferences (cp/infer-batch-sequence context network
                                                           (ds/get-batches dataset batch-size
                                                                           infer-batch-type
                                                                           input-streams)
                                                           {})))))))


(defn inferences->node-id-loss-pairs
  "Given the set of inferences from an inference run of the network
and the set of labels along with the bindings (traverse/get-io-bindings built-network)
return a map of node-id -> loss.  Note that inferences are map of node-id->batches
while labels is a map of stream->data."
  [network inferences dataset-outputs]
  (let [inference-columns (ds/batches->columns inferences)
        label-columns (ds/batches->columns dataset-outputs)
        output-bindings (traverse/get-output-training-bindings network)
        node-id->output-streams (->> output-bindings
                                     (map (fn [{:keys [node-id stream]}]
                                            [node-id stream]))
                                     (into {}))
        ;;inferences are organized by node id
        ;;dataset-outputs are organized by dataset stream
        inference-label-pairs (->> (keys inference-columns)
                                   (map (fn [node-id]
                                          [node-id [(get inference-columns node-id)
                                                    (get label-columns (get node-id->output-streams node-id))]]))
                                   (into {}))]
    (->> output-bindings
         (mapv (fn [{:keys [node-id loss]}]
                 (let [[inferences outputs] (get inference-label-pairs node-id)]
                   [node-id (loss/average-loss loss inferences outputs)]))))))


(defn- setup-network
  "Setup a network for either training or inference."
  [context network input-bindings output-bindings batch-size traverse-fn]
  (as-> (assoc network :batch-size batch-size) network
      (traverse/bind-input-bindings network input-bindings)
      (traverse/bind-output-bindings network output-bindings)
      (traverse-fn network)
      (bind-to-network context network {})))


(defn train
  "Create a sequence of training networks.  This call should be wrapped
in a resource context.  The return value is a lazy sequence of maps with either
just the network for each epoch or the network along with inferences for each
epoch. The inferences are a sequence of maps so if you want just all the inferences
in a single map you still need to call cortex-dataset/batches->columns."
  [context network dataset input-bindings output-bindings
   & {:keys [batch-size infer-batch-type optimiser disable-infer?]
      :or {batch-size 128 infer-batch-type :cross-validation
           optimiser (optimise/adam)}}]
  (let [train-fn (if disable-infer?
                   #(train-seq context % dataset)
                   #(train-infer-seq context % dataset :infer-batch-type infer-batch-type))]
    (-> (setup-network context network input-bindings output-bindings batch-size
                       #(traverse/network->training-traversal % :optimiser optimiser))
      train-fn)))


(defn infer
  "Given a network and a dataset infer a set of data.  data is returned as a sequence of maps of:
node-id->data-stream.  If you want a single map (coalescing all the batches into one item) then
call cortex-dataset/batches->columns"
  [context network dataset input-bindings output-bindings
   & {:keys [batch-size infer-batch-type]
      :or {batch-size 128 infer-batch-type :holdout}}]
  (as-> (setup-network context network input-bindings output-bindings batch-size
                       #(traverse/network->inference-traversal %)) network-or-seq
    (cp/infer-batch-sequence context network-or-seq
                             (ds/get-batches dataset
                                             batch-size
                                             infer-batch-type
                                             (traverse/get-input-streams network-or-seq))
                             {})))


(defn infer-columns
  "Call infer, force realization of everything and return a single map of node-id->output-stream.
This does not need to be wrapped in a resource context; that is done for you."
  [context network dataset input-bindings output-bindings & args]
  (resource/with-resource-context
   (->> (apply infer context network dataset input-bindings output-bindings args)
        ds/batches->columnsv)))
