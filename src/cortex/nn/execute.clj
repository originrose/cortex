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
  (:require
    [clojure.pprint :as pprint]
    [clojure.core.matrix :as m]
    [clojure.set :as c-set]
    [clojure.core.matrix.macros :refer [c-for]]
    [think.resource.core :as resource]
    [think.datatype.core :as dtype]
    [cortex.dataset :as ds]
    [cortex.graph :as graph]
    [cortex.loss :as loss]
    [cortex.util :as util]
    [cortex.optimize :as optimize]
    [cortex.optimize.adam :as adam]
    [cortex.nn.network :as network]
    [cortex.nn.traverse :as traverse]
    [cortex.nn.layers :as layers]
    [cortex.compute.driver :as drv]
    [cortex.compute.math :as math]
    [cortex.compute.loss :as compute-loss]
    [cortex.compute.cpu.backend :as cpu]
    [cortex.compute.nn.layers :as compute-layers]
    [cortex.compute.nn.backend :as backend]
    [cortex.compute.nn.protocols :as compute-protocols]
    [cortex.nn.compute-binding :as compute-binding]))


(defn train-batch!
  [network forward-buffer-map & {:keys [optimize?]
                                 :or [optimize? true]}]
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


(defn generate-numeric-gradients
  "Run network forward and backward like 'forward-backward' but also calculate numeric
  gradients w/r/t the loss function and the provided answer.  This allows for gradient
  checking.  The data should be saved back to the network after the passes."
  [network context batch-size stream->input-map epsilon]
  (resource/with-resource-context
    (let [network (compute-binding/bind-context-to-network network
                                                           context
                                                           batch-size
                                                           (traverse/training-traversal network)
                                                           {:gradients? true
                                                            :numeric-gradients? true})
          ;;Generate all of the calculated gradients.
          parameters (compute-binding/parameters network)
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
                                                                 (compute-binding/datatype network) :training)
          forward-fn (fn [param-value host-buffer device-buffer elem-count idx]
                       (dtype/set-value! host-buffer idx param-value)
                       (drv/copy-host->device stream host-buffer 0 device-buffer 0 elem-count)
                       ;;Raw-forward is used here to avoid calling prepare-forward again.  But this
                       ;;is not an inference pass; it is an actual forward pass.
                       (compute-binding/do-traverse network :raw-forward)
                       (compute-binding/batches->columnsv
                        (compute-binding/output-values network output-buffers)))
          stream->batches-map (->> stream->input-map
                                   (map (fn [[k v]]
                                          [k (->> v
                                                  m/eseq
                                                  (partition (/ (m/ecount v)
                                                                batch-size))
                                                  (mapv vec))]))
                                   (into {}))
          data->loss (fn [inference-data]
                       (compute-binding/execute-live-loss-fn context network
                                                             inference-data
                                                             stream->batches-map))]
      (doseq [{:keys [buffer numeric-gradient host-buffer] :as entry} numeric-buffers]
        (let [device-buffer (math/device-buffer buffer)]
          (when-not (and numeric-gradient host-buffer)
            (throw (ex-info "failed to allocate appropriate buffers for numeric gradients."
                            {:buffer-keys (keys entry)})))
          (let [elem-count (m/ecount buffer)]
            (drv/copy-device->host stream device-buffer 0 host-buffer 0 elem-count)
            (drv/wait-for-event (drv/create-event stream))
            (doseq [idx (range elem-count)]
              (let [param-value (double (dtype/get-value host-buffer idx))
                    positive (forward-fn (+ param-value epsilon) host-buffer device-buffer elem-count idx)
                    negative (forward-fn (- param-value epsilon) host-buffer device-buffer elem-count idx)
                    ;;The loss is normally divided by the batch size to get an average loss
                    ;;but in our case we don't want the average; we want the actual loss.
                    gradient (/ (* (- (double (data->loss positive))
                                      (double (data->loss negative)))
                                   batch-size)
                                (* 2 epsilon))]
                (dtype/set-value! host-buffer idx param-value)
                ;;Reset device buffer to original value.
                (drv/copy-host->device stream host-buffer 0 device-buffer 0 elem-count)
                (dtype/set-value! numeric-gradient idx gradient))))))
      (compute-binding/save-to-network context network {:save-gradients? true}))))



(defn dataset-batches
  "Paritions the dataset into batches and does the seq-of-maps ->
  map-of-seqs transformation."
  [dataset batch-size]
  (let [initial-map (zipmap (keys (first dataset)) (repeat []))]
    (->> dataset
         (partition batch-size)
         (map #(apply merge-with conj initial-map %)))))


(defn- augmented-stream-key?
  [k]
  (and (map? k) (:stream k) (:augmentation k)))


;; TODO: can we get rid of required keys here by pre-filtering the dataset (from the traversal leaves)?
(defn batch-buffers
  [network batch training?]
  (let [driver (compute-binding/driver network)
        stream (compute-binding/stream network)
        datatype (compute-binding/datatype network)
        required-keys (clojure.set/union
                       (if training?
                         (network/graph-streams network :training)
                         (network/graph-streams network :training))
                       (set (filter augmented-stream-key? (keys batch))))
        batch-size (long (:batch-size network))]
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
                     (throw (ex-info "Dataset batch missing key" {:key k})))
                 data-size (long (m/ecount data))
                 item-size (quot data-size batch-size)
                 _ (when-not (= 0 (rem data-size (long batch-size)))
                     (throw (ex-info "Data coming from batch is not multiple of batch-size"
                                     {:data-size data-size
                                      :batch-size batch-size
                                      :stream k})))
                 device-array (math/new-array driver
                                              stream
                                              datatype
                                              [item-size]
                                              batch-size)
                 host-buffer (drv/allocate-host-buffer driver
                                                       (* item-size batch-size)
                                                       datatype)]
             [k {:device-array device-array
                 :host-buffer host-buffer}]))
         (into {}))))


(defn load-batch!
  [network batch batch-buffers]
  (doseq [[k {:keys [device-array host-buffer]}] batch-buffers]
    (let [data (get batch k)
          data (if (map? data) (:data data) data)
          item-count (second (dtype/copy-raw->item! data host-buffer 0))]
      (when-not (= item-count (m/ecount host-buffer))
        (throw (ex-info "Failed to load-batch!"
                        {:item-count item-count
                         :buffer-size (m/ecount host-buffer)}))))
    (drv/copy-host->device (compute-binding/stream network)
                           host-buffer 0
                           (math/device-buffer device-array) 0
                           (m/ecount host-buffer))))


(defn- cuda-backend-fn
  [datatype force-cuda?]
  (fn []
    (try
      (require 'cortex.compute.cuda.backend)
      ((resolve 'cortex.compute.cuda.backend/backend) datatype)
      (catch Throwable e
        (if force-cuda?
          (throw (ex-info "Unable to initialize CUDA back-end for GPU support."
                          {:error e}))
          (do
            (println "CUDA backend creation failed, reverting to CPU")
            (cpu/backend datatype)))))))


(defn compute-context
  "Attempt to create a cuda context, and then only if that fails create a cpu context."
  [& {:keys [datatype backend]
      :or {datatype :float}}]
  (let [cuda-fn (when-not (= backend :cpu)
                  (cuda-backend-fn datatype (= backend :cuda)))]
    {:backend-fn (or cuda-fn #(cpu/backend datatype))
     :datatype datatype}))


(defn train
  [network dataset &
   {:keys [batch-size context optimizer datatype]
    :or {batch-size 10
         datatype :double}}]
  (resource/with-resource-context
    (let [optimizer (or optimizer (adam/adam))
          context (or context (compute-context :datatype datatype))
          network (compute-binding/bind-context-to-network context
                                                     batch-size
                                                     (traverse/training-traversal network)
                                                     {:optimizer optimizer})
          batches (->> (dataset-batches dataset batch-size)
                       (map (partial graph/augment-streams (network/network->graph network))))
          batch-buffers (batch-buffers network (first batches) true)
          stream (compute-binding/stream network)
          stream->buffer-map (zipmap (keys batch-buffers)
                                     (map :device-array (vals batch-buffers)))
          network (assoc-in network [:compute-binding :stream->buffer-map]
                            stream->buffer-map)]
      (doseq [batch batches]
        (load-batch! network batch batch-buffers)
        (train-batch! network stream->buffer-map :optimize? true))
      (compute-binding/save-to-network context network {}))))


(defn run
  "Run a network on a dataset.  The results are returned as a sequence of
  maps where the node :id is the key for each output value."
  [network dataset & {:keys [batch-size context datatype]
                      :or {batch-size 1
                           datatype :double}
                      :as options}]
  (resource/with-resource-context
    (let [context (or context (compute-context :datatype datatype))
          ;;In the case where the context was passed in we ignore the datatype argument
          ;;else we run into problems pulling data off the gpu.
          datatype (get context :datatype)
          network (compute-binding/bind-context-to-network context
                                                     batch-size
                                                     (traverse/inference-traversal network) {})
          batches (->> (dataset-batches dataset batch-size)
                       (map (partial graph/augment-streams (network/network->graph network))))
          _ (when (empty? batches)
              (throw (ex-info "Batches were empty, perhaps batch-size > (count dataset)?")))
          batch-buffers (batch-buffers network (first batches) false)
          stream->buffer-map (zipmap (keys batch-buffers)
                                     (map :device-array (vals batch-buffers)))
          network (assoc-in network
                            [:compute-binding :stream->buffer-map]
                            stream->buffer-map)
          output-buffers (compute-binding/output-binding-buffers network batch-size datatype)]
      (reduce
       (fn [results next-batch]
         (load-batch! network next-batch batch-buffers)
         ;; (println "\nTraversal:")
         ;; (clojure.pprint/pprint (get-in network [:traversal :forward]))
         (compute-binding/do-traverse network stream->buffer-map :inference)
         (concat results (compute-binding/output-values network output-buffers)))
       []
       batches))))
