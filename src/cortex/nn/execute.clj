(ns cortex.nn.execute
  "Executing the graph means training or inference.  The goal is to allow both
imperative/effectful implementations and pure functional implementations but to abstract
common details of training or execution into one place written in such a way that someone
can affect the behavior of various implementations and design new execution strategies
(like parameter sharing) at least partially without needing to work withing a specific
implementation.  It is important to realize that training the network means essentially
a transformation from compute-graph -> compute-graph via some training process.
Both train and infer should be wrapped in resource contexts; this is not done at this level.
Furthermore infer should be both wrapped in a resource context and completely realized."
  (:require [clojure.pprint :as pprint]
            [clojure.core.matrix :as m]
            [clojure.set :as c-set]
            [clojure.java.io :as io]
            [clojure.core.matrix.macros :refer [c-for]]
            [think.resource.core :as resource]
            [think.datatype.core :as dtype]
            [think.datatype.base :as dtype-base]
            [think.parallel.core :as parallel]
            [cortex.graph :as graph]
            [cortex.loss.core :as loss]
            [cortex.util :as util]
            [cortex.optimize :as optimize]
            [cortex.optimize.adam :as adam]
            [cortex.nn.network :as network]
            [cortex.nn.traverse :as traverse]
            [cortex.nn.layers :as layers]
            [cortex.compute.driver :as drv]
            [cortex.compute.math :as math]
            [cortex.compute.cpu.backend :as cpu]
            [cortex.compute.nn.layers :as compute-layers]
            [cortex.compute.nn.backend :as backend]
            [cortex.compute.nn.protocols :as compute-protocols]
            [cortex.nn.compute-binding :as compute-binding]))


(defn- cuda-backend-fn
  [datatype force-cuda?]
  (fn []
    (try
      (require 'cortex.compute.cuda.backend)
      (if-let [backend-fn (resolve 'cortex.compute.cuda.backend/backend)]
        (backend-fn :datatype datatype)
        (throw (ex-info "cortex.compute.cuda.backend function compilation failed. Please ensure that CudNN is installed." {})))
      (catch Throwable e
        (if force-cuda?
          (throw (ex-info "Unable to initialize CUDA back-end for GPU support."
                          {:error e}))
          (do
            (let [error-path (str (System/getProperty "user.home") "/.cortex/last-error")]
              (io/make-parents error-path)
              (spit error-path (with-out-str (pprint/pprint {:message (.getMessage e)
                                                             :str (str e)
                                                             :data (ex-data e)
                                                             :stacktrace (map str (.getStackTrace e))}))))
            (println "CUDA backend creation failed, reverting to CPU")
            (cpu/backend :datatype datatype)))))))


(defn compute-context
  "Attempt to create a cuda context, and then only if that fails create a cpu context."
  [& {:keys [datatype backend]
      :or {datatype :float}}]
  (let [cuda-fn (when-not (= backend :cpu)
                  (cuda-backend-fn datatype (= backend :cuda)))]
    {:backend-fn (or cuda-fn #(cpu/backend :datatype datatype))
     :datatype datatype}))


(def ^:dynamic *current-backend* nil)


(defmacro with-compute-context
  [context & body]
  `(resource/with-resource-context
     ;;Avoid creating multiple backends when we can; this is a significant performance and
     ;;memory issue that users should be able to avoid.
     (let [backend# (or *current-backend* ((get ~context :backend-fn)))]
       (with-bindings {#'*current-backend* backend#}
         (backend/with-backend backend#
           ~@body)))))

(defn current-backend
  []
  (when-not *current-backend*
    (throw (ex-info "No current backend bound.  Call with-compute-context." {})))
  *current-backend*)



(defn- normalize-argument-buffer
  [arg-buf]
  (let [buf-value (get arg-buf :buffer)]
    (if (map? buf-value)
      (assoc arg-buf :buffer (get buf-value :data))
      arg-buf)))


(defn- execute-loss-term
  "Execute a loss term.  This uses the context to find node and loss parameters."
  [graph loss-term inference-maps dataset-maps]
  (* (double (loss/get-loss-lambda loss-term))
     (/ (->> (map #(->> (graph/resolve-arguments graph loss-term %1 %2)
                        (loss/loss loss-term))
                  dataset-maps
                  inference-maps)
             (apply +))
        (count inference-maps))))


(defn live-parameter-graph
  [network]
  (-> (network/network->graph network)
      (assoc :buffers #(compute-binding/get-parameter network %))))


(defn execute-bound-loss-fn
  "Execute a loss function against a running network returning the loss value as a double.
  Inferences and dataset outputs are expected to be maps of data."
  [network inferences dataset-outputs]
  (let [param-graph (live-parameter-graph network)]
   (apply + (->> (network/loss-function network)
                 (map #(execute-loss-term param-graph % inferences dataset-outputs))))))



(defn- augment-and-normalize-streams
  [graph batch-data]
  (->> (graph/augment-streams graph batch-data)
       (map (fn [[k v]]
              [k (if (map? v)
                   (get v :data)
                   v)]))
       (into {})))


(defn execute-loss-fn
  "Given the set of inferences from an inference run of the network and the set of labels along
  with the bindings (traverse/get-io-bindings network) return the loss function from the
  traverse where each term has a :value member with it's post-lambda-multiplied value."
  [network inferences dataset]
  (let [augmented-dataset (->> dataset
                               compute-binding/batches->columns
                               (augment-and-normalize-streams (network/network->graph network))
                               compute-binding/columns->maps)
        ;;In this case we assum the graph has updated versions of the parameters
        ;;So we map to a function that returns exactly the parameter.
        param-graph (-> (network/network->graph network)
                        (assoc :buffers #(get-in network [:compute-graph :buffers % :buffer])))]
    (->> (network/loss-function network)
         (mapv (fn [loss-term]
                 (->> (execute-loss-term param-graph loss-term
                                         inferences augmented-dataset)
                      (assoc loss-term :value)))))))


(defn train-batch!
  [network forward-buffer-map & {:keys [optimize?]
                                 :or {optimize? true}}]
  (-> network
      (compute-binding/update-traversal-buffers forward-buffer-map :stream :buffer)
      (compute-binding/do-traverse :forward)
      (compute-binding/zero-traverse-gradients)
      (compute-binding/compute-loss-term-gradients)
      (compute-binding/do-traverse :backward)
      (#(if optimize?
          (compute-binding/optimize-network %)
          %)))
  :ok)


(defn- print-time-info
  [name]
  #_(println (format "%s - %s - %s" name (.getId (Thread/currentThread)) (System/currentTimeMillis))))


(defn dataset-batches
  "Paritions the dataset into batches and does the seq-of-maps ->
  map-of-seqs transformation."
  [dataset batch-size]
  (let [initial-map (zipmap (keys (first dataset)) (repeat []))]
    (->> dataset
         (partition batch-size)
         (map #(do
                 (print-time-info :dataset-batches-start)
                 (let [retval (apply merge-with conj initial-map %)]
                   (print-time-info :dataset-batches-end)
                   retval))))))

(defn- augment-streams
  [network dataset batch-size]
  (let [graph-augmentations (graph/get-stream-augmentation-arguments (network/network->graph network))
        batches (->> (dataset-batches dataset batch-size)
                     (map (fn [item]
                            (print-time-info :augment-streams-start)
                            (let [retval
                                  (graph/perform-stream-augmentations graph-augmentations item)]
                              (print-time-info :augment-streams-end)
                              retval))))
        _ (when (empty? batches)
            (throw (ex-info "Batches were empty, perhaps batch-size > (count dataset)?"
                            {:batch-size batch-size
                             :dataset-count (count dataset)})))]
    batches))


(defn- update-traversal
  [network traversal batches]
  (let [first-batch (first batches)]
    (update traversal :buffers
            (fn [buffer-map]
              (->> buffer-map
                   (map (fn [[k v]]
                          ;;Is this an augmented buffer
                          (if (get-in k [:stream :augmentation])
                            (let [aug-key (get k :stream)
                                  batch-entry (get first-batch aug-key)]
                              (when-not batch-entry
                                (throw (ex-info "Failed to find dataset element for augmented key:"
                                                {:aug-key aug-key
                                                 :dataset-keys (keys first-batch)})))
                              (let [elem-size (-> (if (map? batch-entry)
                                                    (get batch-entry :data)
                                                    batch-entry)
                                                  first
                                                  m/ecount)]
                                [k (assoc v :dimension {:width elem-size})]))
                            [k v])))
                   (into {}))))))


(defn- augment-streams-and-update-traversal
  "Use the graph to augment any incoming streams.  Then use the first item from the batch to get the
element count from the augmented stream and update traversal buffers so their dimensions match the
augmented element count for the traversal buffers that pertain augmented streams.  This is required
because at this point augmented streams do not need to specify a shape, so the shape must be derived
from the size of their result and the traversal information updated to take this into account."
  [network traversal dataset batch-size]
  (let [batches (augment-streams network dataset batch-size)
        traversal (update-traversal network traversal batches)]
    {:traversal traversal
     :batches batches}))


(defn generate-numeric-gradients
  "Run network forward and backward like 'forward-backward' but also calculate numeric
  gradients w/r/t the loss function and the provided answer.  This allows for gradient
  checking.  The data should be saved back to the network after the passes."
  [network context batch-size dataset epsilon]
  (with-compute-context context
    (let [{:keys [traversal batches]} (augment-streams-and-update-traversal
                                       network
                                       (traverse/training-traversal network
                                                                    :keep-non-trainable? true)
                                       dataset
                                       batch-size)
          network (compute-binding/bind-context-to-network
                   network
                   (current-backend)
                   batch-size
                   traversal
                   ;;A lot of the gradient tests have no trainable nodes so we have to disable
                   ;;the backward pass optimization where we do not traverse nodes that contribute
                   ;;no useful gradients to the solution.
                   {:gradients? true
                    :numeric-gradients? true})
          ;;Generate all of the calculated gradients.
          parameters (compute-binding/parameters network)
          ;;The first batch is the stream->input-map.  But now it is augmented.
          stream->input-map (first batches)
          ;;Store the input buffers as traversal buffers
          network (compute-binding/update-traversal-buffers
                   network
                   (compute-binding/load-id->input-map network stream->input-map)
                   :stream
                   :buffer)
          ;;This calls prepare-forward exactly once and does one forward
          ;;plus backward and loss gradient to generate calculated gradients
          _ (train-batch! network {} :optimize? false)
          ;;generate a sequence of buffers where we will generate numeric gradients for each buffer.
          ;;We use the inference graph because we want to check input gradients.
          numeric-buffers (concat (->> (network/graph-streams network :inference)
                                       (map (fn [[stream dims]]
                                              (let [map-key {:stream stream}]
                                                (merge map-key
                                                       (compute-binding/find-traversal-buffer
                                                        network
                                                        map-key))))))
                                  (filter #(get % :gradients?) parameters))
          epsilon (double epsilon)
          stream (compute-binding/stream network)
          batch-size (compute-binding/batch-size network)
          output-buffers (compute-binding/output-binding-buffers network batch-size
                                                                 (compute-binding/datatype network)
                                                                 :training)
          stream->batches-map (->> stream->input-map
                                   (map (fn [[k v]]
                                          [k (->> v
                                                  m/eseq
                                                  (partition (/ (m/ecount v)
                                                                batch-size))
                                                  (mapv vec))]))
                                   (into {}))
          stream-maps (-> stream->batches-map
                          compute-binding/columns->maps)
          ;;Run the network forward and generate the loss.
          forward-fn (fn [param-value host-buffer device-buffer elem-count idx]
                       (dtype-base/set-value! host-buffer idx param-value)
                       (drv/copy-host->device stream host-buffer 0 device-buffer 0 elem-count)
                       ;;Raw-forward is used here to avoid calling prepare-forward again.  But this
                       ;;is not an inference pass; it is an actual forward pass.
                       (compute-binding/do-traverse network :raw-forward)
                       (let [net-outputs (compute-binding/output-values network output-buffers)]
                         (execute-bound-loss-fn
                          network
                          net-outputs
                          stream-maps)))]
      (doseq [{:keys [buffer numeric-gradient host-buffer] :as entry} numeric-buffers]
        (let [device-buffer (math/device-buffer buffer)]
          (when-not (and numeric-gradient host-buffer)
            (throw (ex-info "failed to allocate appropriate buffers for numeric gradients."
                            {:buffer-keys (keys entry)
                             :entry entry})))
          (let [elem-count (m/ecount buffer)]
            (drv/copy-device->host stream device-buffer 0 host-buffer 0 elem-count)
            (drv/sync-stream stream)
            (doseq [idx (range elem-count)]
              (let [param-value (double (dtype-base/get-value host-buffer idx))
                    positive (forward-fn (+ param-value epsilon) host-buffer device-buffer elem-count idx)
                    negative (forward-fn (- param-value epsilon) host-buffer device-buffer elem-count idx)
                    ;;The loss is normally divided by the batch size to get an average loss
                    ;;but in our case we don't want the average; we want the actual loss.
                    gradient (/ (* (- (double positive)
                                      (double negative))
                                   batch-size)
                                (* 2 epsilon))]
                (dtype-base/set-value! host-buffer idx param-value)
                ;;Reset device buffer to original value.
                (drv/copy-host->device stream host-buffer 0 device-buffer 0 elem-count)
                (dtype-base/set-value! numeric-gradient idx gradient))))))
      (-> (compute-binding/save-to-network context network {:save-gradients? true})
          :network))))



(defn- augmented-stream-key?
  [k]
  (and (map? k) (:stream k) (:augmentation k)))


;; TODO: can we get rid of required keys here by pre-filtering the dataset (from the traversal leaves)?
(defn batch-buffers
  [stream network batch training?]
  (let [datatype (compute-binding/datatype network)
        required-keys (clojure.set/union
                       (->> (if training?
                              (network/graph-streams network :training)
                              (network/graph-streams network :inference))
                            (map first)
                            set)
                       (set (filter augmented-stream-key? (keys batch))))
        batch-size (compute-binding/batch-size network)]
    (when (zero? (count required-keys))
      (throw (ex-info "Zero required keys in batch-buffers" {})))
    (->> (for [k required-keys]
           (let [[data datatype] (if (map? k)
                                   (let [augmented-stream-val (get batch k)]
                                     (if (map? augmented-stream-val)
                                       [(:data augmented-stream-val) (:datatype augmented-stream-val)]
                                       [augmented-stream-val datatype]))
                                   [(get batch k) datatype])
                 _ (when (nil? data)
                     (throw (ex-info "Dataset batch missing key"
                                     {:key k
                                      :dataset-keys (keys batch)})))
                 data-size (long (m/ecount data))
                 item-size (quot data-size batch-size)
                 _ (when-not (= 0 (rem data-size (long batch-size)))
                     (throw (ex-info "Data coming from batch is not multiple of batch-size"
                                     {:data-size data-size
                                      :batch-size batch-size
                                      :stream k})))
                 device-array (math/new-array stream
                                              datatype
                                              [item-size]
                                              batch-size)
                 host-buffer (drv/allocate-host-buffer (drv/get-driver stream)
                                                       (* item-size batch-size)
                                                       datatype
                                                       :usage-type :reusable)]
             [k {:device-array device-array
                 :host-buffer host-buffer}]))
         (into {}))))


(defn load-batch!
  [stream batch batch-buffers]
  (doseq [[k {:keys [device-array host-buffer]}] batch-buffers]
    (let [data (get batch k)
          data (if (map? data) (:data data) data)
          item-count (second (dtype/copy-raw->item! data host-buffer 0))]
      (when-not (= item-count (m/ecount host-buffer))
        (throw (ex-info "Failed to load-batch!"
                        {:item-count item-count
                         :buffer-size (m/ecount host-buffer)}))))
    (drv/copy-host->device stream
                           host-buffer 0
                           (math/device-buffer device-array) 0
                           (m/ecount host-buffer))))


(defn- dataset->uploading-batches
  [network batches batch-transfer-parallelism training?]
  (let [batch-transfer-parallelism (long (max batch-transfer-parallelism 1))
        device (drv/current-device)
        batch-buffer-seq (->> (range batch-transfer-parallelism)
                              (mapv (fn [_]
                                      (let [stream (drv/create-stream)
                                            batch-buffers (batch-buffers stream network (first batches) training?)]
                                        {:batch-buffers batch-buffers
                                         :stream->buffer-map (zipmap (keys batch-buffers)
                                                                     (map :device-array (vals batch-buffers)))
                                         :stream stream}))))]
    (->> (map (fn [batch batch-buffer]
                (assoc batch-buffer :batch batch))
              batches (->> (repeat batch-buffer-seq)
                           (apply concat)))
         (parallel/queued-pmap (- batch-transfer-parallelism 1)
                               (fn [{:keys [batch-buffers stream->buffer-map stream batch]}]
                                 (print-time-info :upload-batch-start)
                                 (load-batch! stream batch batch-buffers)
                                 (print-time-info :upload-batch-end)
                                 {:stream->buffer-map stream->buffer-map
                                  :batch-stream stream})))))

(defn train
  "Train.  Returns a tuple of network and optimizer where both the network and optimizer's
  parameters are updated."
  [network dataset & {:keys [batch-size context optimizer datatype batch-transfer-parallelism
                             save-gradients?]
                      :or {batch-size 10
                           datatype :float
                           batch-transfer-parallelism 2
                           save-gradients? false}}]
  (let [context (or context (compute-context :datatype datatype))]
    (with-compute-context context
      (let [optimizer (or optimizer (adam/adam))
            batches (augment-streams network dataset batch-size)

            traversal (update-traversal network
                                        (traverse/training-traversal network)
                                        batches)
            network (compute-binding/bind-context-to-network
                     network
                     (current-backend)
                     batch-size
                     traversal
                     {:optimizer optimizer})]
        (doseq [{:keys [stream->buffer-map batch-stream]}
                (dataset->uploading-batches network batches
                                            batch-transfer-parallelism
                                            true)]
          (print-time-info :train-start)
          ;;Ensure the data is uploaded
          (drv/sync-streams batch-stream (compute-binding/stream network))
          (train-batch! network stream->buffer-map :optimize? true)
          ;;Ensure the network is finished before we upload more things.
          (drv/sync-streams (compute-binding/stream network) batch-stream)
          (print-time-info :train-end))
        (compute-binding/save-to-network context network
                                         {:save-gradients? save-gradients?
                                          :save-optimizer-parameters? true})))))


(defn run
  "Run a network on a dataset.  The results are returned as a sequence of maps where the node
  :id is the key for each output value.  There is an option to include outputs required to
  generate the actual network loss."
  [network dataset & {:keys [batch-size context datatype loss-outputs? batch-transfer-parallelism]
                      :or {batch-size 1
                           datatype :float
                           batch-transfer-parallelism 2}
                      :as options}]
  (let [context (or context (compute-context :datatype datatype))]
    (with-compute-context context
      (let [batches (augment-streams network dataset batch-size)
            traversal (update-traversal network
                                        (traverse/inference-traversal network)
                                        batches)
            network (compute-binding/bind-context-to-network
                     network
                     (current-backend)
                     batch-size
                     traversal
                     {})
            datatype (dtype/get-datatype (current-backend))
            output-buffers (compute-binding/output-binding-buffers network
                                                                   batch-size
                                                                   datatype
                                                                   (if loss-outputs?
                                                                     :training
                                                                     :inference))]
        (->> (dataset->uploading-batches network batches
                                         batch-transfer-parallelism
                                         false)
             (mapcat (fn [{:keys [stream->buffer-map batch-stream]}]
                       ;;Ensure data has finished before going on to next thing
                       (drv/sync-streams batch-stream (compute-binding/stream network))
                       (let [network (compute-binding/update-traversal-buffers network stream->buffer-map
                                                                               :stream :buffer)]
                         (compute-binding/do-traverse network :inference)
                         ;;output-values has an assumed sync at this point.
                         (let [retval (compute-binding/output-values network output-buffers)]
                           ;;Ensure whatever output path is necessary has finished before we start uploading
                           ;;more data.
                           (drv/sync-streams (compute-binding/stream network) batch-stream)
                           retval))))
             vec)))))
