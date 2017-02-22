(ns cortex.nn.execute
  "Executing the graph means training or inference.  The goal is to allow both
imperative/effectful implementations and pure functional implementations but to abstract
common details of training or execution into one place written in such a way that someone
can affect the behavior of various implementations and design new execution strategies
(like parameter sharing) at least partially without needing to work withing a specific
implementation.  It is important to realize that training the network means essentially
a transformation from layer-graph -> layer-graph via some training process.
Both train and infer should be wrapped in resource contexts; this is not done at this level.
Furthermore infer should be both wrapped in a resource context and completely realized."
  (:require [cortex.nn.traverse :as traverse]
            [cortex.nn.network :as network]
            [cortex.nn.layers :as layers]
            [cortex.dataset :as ds]
            [think.resource.core :as resource]
            [cortex.loss :as loss]
            [cortex.nn.protocols :as cp]
            [cortex.optimize :as optimize]
            [cortex.optimize.adam :as adam]
            [clojure.pprint :as pprint]
            [cortex.graph :as graph]))

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
  (cp/save-to-network context network options))


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

(defn- map-val->map-buffers
  [map-data]
  (->> map-data
       (map (fn [[k v]]
              [k {:buffer v}]))
       (into {})))


(defn- normalize-argument-buffer
  [arg-buf]
  (let [buf-value (get arg-buf :buffer)]
    (if (map? buf-value)
      (assoc arg-buf :buffer (get buf-value :data))
      arg-buf)))


(defn- execute-live-loss-term
  "Execute a loss term.  This uses the context to find node and loss parameters."
  [context network loss-term inference-columns dataset-columns]
  (let [graph (-> (network/network->graph network)
                  (assoc :buffers #(hash-map :buffer
                                             [(cp/get-parameter context network %)])))
        arguments (->> (graph/resolve-arguments graph loss-term
                                                (map-val->map-buffers dataset-columns)
                                                (map-val->map-buffers inference-columns))
                       (map (fn [[k v]]
                              (let [v (normalize-argument-buffer v)]
                               (try
                                 [k (assoc v
                                           :count
                                           (count (get v :buffer)))]
                                 (catch Throwable e
                                   (throw (ex-info "Argument resolved to odd value"
                                                   {:arg-key k
                                                    :error e})))))))
                       (into {}))
        distinct-count (->> arguments
                            (map (comp :count second))
                            distinct)
        _ (when-not (< (count distinct-count)
                       3)
            (throw (ex-info "There should be at most 2 distinct argument buffer counts"
                            {:buffer-counts (map (fn [[k v]]
                                                   [k
                                                    (dissoc v :buffer)])
                                                 arguments)})))
        max-argument-num-items (apply max distinct-count)
        even-arguments (->> arguments
                            (map (fn [[k argument]]
                                   [k
                                    (update argument :buffer
                                            (fn [buffer]
                                              (->> (repeat buffer)
                                                   (apply concat)
                                                   (take max-argument-num-items)
                                                   vec)))])))
        argument-keys (map first arguments)
        argument-vals (map second arguments)
        partitioned-buffers (->> argument-vals
                                 (map :buffer)
                                 (apply interleave)
                                 (partition (count even-arguments)))
        buffer-map-seq (map (fn [key-seq buf-seq]
                              (->> (map vector key-seq buf-seq)
                                   (into {})))
                            (repeat argument-keys) partitioned-buffers)]
    (* (double (loss/get-loss-lambda loss-term))
       (/ (->> buffer-map-seq
               (map #(loss/loss loss-term %))
               (apply +))
          (count buffer-map-seq)))))


(defn loss-fn->table-str
  [loss-fn]
  (with-out-str
    (pprint/print-table [:type :value :lambda :node-id :argument]
                        (mapv (fn [loss-term]
                                (assoc loss-term
                                       :lambda
                                       (loss/get-loss-lambda
                                        loss-term)
                                       :node-id
                                       (get-in loss-term
                                               [:output
                                                :node-id])
                                       :argument
                                       (get-in loss-term
                                               [:output
                                                :argument])))
                              loss-fn))))


(defn execute-live-loss-fn
  "Execute a loss function against a running network returning the loss value as a double.  Inferences
and dataset outputs are expected to be maps of columns of data."
  [context network inferences dataset-outputs]
  (apply + (->> (get-in network [:traversal :loss-function])
                (map #(execute-live-loss-term context network % inferences dataset-outputs)))))


(defn- augment-and-normalize-streams
  [graph batch-data]
  (->> (graph/augment-streams graph batch-data)
       (map (fn [[k v]]
              [k (if (map? v)
                   (get v :data)
                   v)]))
       (into {})))


(defn network->applied-loss-fn
  "Given the set of inferences from an inference run of the network
and the set of labels along with the bindings (traverse/get-io-bindings built-network)
return the loss function from the traverse where each term has a :value member with it's
post-lambda-multiplied value."
  [context network inferences dataset-outputs]
  (let [inference-columns (ds/batches->columns inferences)
        label-columns (->> dataset-outputs
                           (map #(augment-and-normalize-streams
                                  (network/network->graph network)
                                  %))
                           ds/batches->columns)
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
                                                    (get label-columns
                                                         (get node-id->output-streams
                                                              node-id))]]))
                                   (into {}))]
    (->> (get-in network [:traversal :loss-function])
         (mapv (fn [loss-term]
                 (assoc loss-term
                        :value
                        (execute-live-loss-term context network loss-term
                                                inference-columns label-columns)))))))


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
   & {:keys [batch-size infer-batch-type optimizer disable-infer?]
      :or {batch-size 128 infer-batch-type :cross-validation
           optimizer (adam/adam)}}]
  (let [train-fn (if disable-infer?
                   #(train-seq context % dataset)
                   #(train-infer-seq context % dataset :infer-batch-type infer-batch-type))]
    (-> (setup-network context network input-bindings output-bindings batch-size
                       #(traverse/network->training-traversal
                         %
                         (ds/dataset->stream->size-map dataset)
                         :optimizer optimizer))
      train-fn)))


(defn infer
  "Given a network and a dataset infer a set of data.  data is returned as a sequence of maps of:
node-id->data-stream.  If you want a single map (coalescing all the batches into one item) then
call cortex-dataset/batches->columns"
  [context network dataset input-bindings output-bindings
   & {:keys [batch-size infer-batch-type]
      :or {batch-size 128 infer-batch-type :holdout}}]
  (as-> (setup-network context network input-bindings output-bindings batch-size
                       #(traverse/network->inference-traversal
                         % (ds/dataset->stream->size-map dataset))) network-or-seq
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
