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
  (:require
    [clojure.pprint :as pprint]
    [think.resource.core :as resource]
    [cortex.dataset :as ds]
    [cortex.graph :as graph]
    [cortex.optimize :as optimize]
    [cortex.optimize.adam :as adam]
    [cortex.nn.traverse :as traverse]
    [cortex.nn.network :as network]
    [cortex.nn.layers :as layers]
    [cortex.nn.protocols :refer :all]
    [cortex.compute.cpu.backend :as cpu-backend]
    [cortex.compute.nn.compute-execute :as compute-execute]))


(defn- safe-inc
  [num-or-nil]
  (if (nil? num-or-nil)
    1
    (inc num-or-nil)))


(defn- train-seq
  "Infinite sequence of networks, one for each epoch.
The context is expected to already be bound to the network."
  [context {:keys [batch-size] :as built-network} dataset]
  (let [streams (->> (map :stream (traverse/get-io-bindings built-network))
                             (remove nil?)
                             set)
        dataset-epoch (ds/get-batches dataset batch-size :training streams)
        trained-network (-> (train-batch-sequence context built-network dataset-epoch {})
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
                       :inferences (infer-batch-sequence context network
                                      (ds/get-batches dataset batch-size
                                                      infer-batch-type input-streams)
                                      {})))))))


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
                        (compute-execute/execute-live-loss-term context network loss-term
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
    (infer-batch-sequence context network-or-seq
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

(defn- try-cuda-backend
  [datatype force-cuda?]
  (try
    (require 'cortex.compute.cuda.backend)
    #((resolve 'cortex.compute.cuda.backend/create-backend) datatype)
    (catch Exception e
      (if force-cuda?
        (throw (ex-info "Unable to initialize CUDA back-end for GPU support."
                        {:error e}))
        false))))

(defn create-context
  "Attempt to create a cuda context, and then only if that fails create a cpu context."
  [& {:keys [datatype backend]
      :or {datatype :float}}]
  (let [datatype (or datatype :float)
        cuda-fn (if (= backend :cpu)
                  false
                  (try-cuda-backend datatype (= backend :cuda)))
        config (if cuda-fn
                 {:backend :cuda
                  :backend-fn cuda-fn}
                 {:backend :cpu
                  :backend-fn #(cpu-backend/create-backend datatype)})
        config (assoc config :datatype datatype)]
    (compute-execute/map->ComputeExecutionContext config)))

(defn run
  "Run a network on a dataset.  data is returned as a sequence of maps of:
node-id->data-stream.  If you want a single map (coalescing all the batches into one item) then
call cortex-dataset/batches->columns"
  [network dataset
   & {:keys [batch-size infer-batch-type datatype]
      :or {batch-size 128 infer-batch-type :holdout}
      :as options}]

  (resource/with-resource-context
    (let [context (create-context)
          ; Creates a map of {:<stream-name> {:channel-count c :width w :height h}
          stream-map (ds/dataset->stream->size-map dataset)

          ; convert from vector to graph description if needed
          network (if (and (map? network) (:layer-graph network))
                    network
                    (network/build-network network))
          network (-> network
                      ; set the batch-size
                      (assoc :batch-size batch-size)

                      ; Bind graph nodes to stream names based on their node-id
                      traverse/bind-vars-to-network

                      ; Adds a :traversal map to the network with :forward and
                      ; :backward lists, :buffers, :type, :optimizer, and
                      ; :loss-function keys.
                      ; TODO: change to add inference traversal
                      (traverse/network->inference-traversal stream-map))
          ; Connect the execution context to the network so it can setup any
          ; backend specific data structures or initialization.
          network (bind-to-network context network {})

          ; Get the list of input streams required for the network
          input-streams (traverse/get-input-streams network)

          ; Get a lazy seq of batches
          batches (ds/get-batches dataset
                                  batch-size
                                  infer-batch-type
                                  input-streams)

          ; Plug the data through the model.
          ; NOTE: the doall must be here otherwise everything will get
          ; deallocated when leaving the current resource context!!!
          results (doall (infer-batch-sequence context network batches {}))]
      results)))

