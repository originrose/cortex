(ns cortex-gpu.nn.batch
  (:require [cortex-gpu.nn.cudnn :as cudnn]
            [cortex-gpu.nn.layers :as layers]
            [clojure.core.matrix :as m]
            [cortex-gpu.cuda :as cuda]
            [cortex.dataset :as ds])
  (:import [org.bytedeco.javacpp DoublePointer IntPointer]
           [java.nio DoubleBuffer]
           [cortex_gpu.cuda DevicePointer]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn load-dataset-to-gpu
  "Load the entire dataset to the GPU in one go."
  [dataset]
  (let [data (first dataset)
        labels (second dataset)]
    (if labels
      [(cudnn/vec-of-matrixes data)
       (cudnn/vec-of-matrixes labels)]
      [(cudnn/vec-of-matrixes data)])))


(defn allocate-batch-index-buffer
  [network ^long dataset-count]
  (let [item-byte-size (* dataset-count Integer/BYTES)
        batch-indexes {:host-ptr (IntPointer. (int dataset-count))
                       :device-ptr (cuda/mem-alloc item-byte-size (IntPointer.))}]
    (assoc network :batch-indexes batch-indexes)))


(defn upload-indexes
  [network ^ints indexes]
  (let [dataset-count (alength indexes)
        item-byte-size (* dataset-count Integer/BYTES)
        {:keys [^IntPointer host-ptr ^DevicePointer device-ptr]} (:batch-indexes network)]
    (.put host-ptr indexes)
    (.position ^IntPointer (.ptr device-ptr) 0)
    (cuda/mem-copy-host->device host-ptr device-ptr item-byte-size)
    network))

(defn seq-to-partitioned-vec
  [data-seq batch-size]
  (mapv vec (partition batch-size data-seq)))


(defn clipped-index-count
  ^long [^long dataset-count ^long batch-size]
  (* batch-size (long (/ dataset-count batch-size))))


(defn upload-randomized-indexes
  [network ^long dataset-count batch-size]
  (let [index-data (take (clipped-index-count dataset-count batch-size) (shuffle (range dataset-count)))]
    (assoc (upload-indexes network (int-array index-data))
           :indexes (seq-to-partitioned-vec index-data batch-size))))


(defn upload-sequential-indexes
  [network ^long dataset-count batch-size]
  (let [index-data (take (clipped-index-count dataset-count batch-size) (range dataset-count))]
    (assoc (upload-indexes network (int-array index-data))
           :indexes (seq-to-partitioned-vec index-data batch-size))) )


(defn setup-batch-buffer
  [network batch-size column-count buffer-keyword]
  (let [gpu-batch-buffers (if (sequential? column-count)
                            (mapv #(cudnn/new-array [%] batch-size) column-count)
                            (cudnn/new-array [column-count] batch-size))]
    (assoc network buffer-keyword gpu-batch-buffers)))


(defn indexed-load-buffer
  [batch-indexes gpu-data batch-index batch-buffer]
  (let [[batch-size stride] (cudnn/batch-shape batch-buffer)
        stride (long stride)
        batch-size (long batch-size)
        batch-index (long batch-index)
        index-position (* batch-size batch-index)
        ^DevicePointer index-ptr (:device-ptr batch-indexes)
        _ (.position ^IntPointer (.ptr index-ptr) index-position)]
    (cudnn/indexed-assign gpu-data batch-buffer stride index-ptr batch-size)))


(defn load-batch-buffer
  "Perform indexed copies from the gpu data into a per-batch buffers.
Note that gpu-data and the batch-buffer may be vectors of matrixes as opposed
to a single matrix"
  [{:keys [batch-indexes] :as network} gpu-data batch-index batch-buffer]
  (if (sequential? gpu-data)
    (doall (map (fn [gpu-item batch-buf]
                  (indexed-load-buffer batch-indexes gpu-item batch-index batch-buf))
                gpu-data batch-buffer))
    (indexed-load-buffer batch-indexes gpu-data batch-index batch-buffer))
  network)


(def data-keyword :data-input)
(def label-keyword :label-output)


(defn setup
  [train-config batch-size total-input-count n-input-vec n-output-vec]
  (-> train-config
      (allocate-batch-index-buffer total-input-count)
      (setup-batch-buffer batch-size n-input-vec data-keyword)
      (setup-batch-buffer batch-size n-output-vec label-keyword)))


(def batch-types
  [:training :testing :running])

(defprotocol PBatchingSystem
  ;;return train-config
  ;;Overall setup, called once
  (bs-setup [bs train-config batch-type])
  ;;Per-epoch setup called before each epoch
  (bs-setup-epoch [bs train-config batch-type])
  ;;Per batch setup, called every batch
  (bs-get-buffers [bs train-config batch-idx batch-type])
  (bs-has-cpu-labels? [bs train-config batch-type])
  (bs-get-cpu-labels [bs train-config batch-type]))


(defn setup-batching-system [train-config batch-type]
  (bs-setup (:batching-system train-config) train-config batch-type))

(defn setup-batching-system-per-epoch [train-config batch-type]
  (bs-setup-epoch (:batching-system train-config) train-config batch-type))

(defn get-batching-system-buffers [train-config batch-idx batch-type]
  (bs-get-buffers (:batching-system train-config) train-config batch-idx batch-type))

(defn has-cpu-labels? [train-config batch-type]
  (bs-has-cpu-labels? (:batching-system train-config) train-config batch-type))

(defn get-cpu-labels [train-config batch-type]
  (bs-get-cpu-labels (:batching-system train-config) train-config batch-type))

(defn get-total-input-count
  ^long [train-config]
  (first (cudnn/shape (ffirst (get train-config :gpu-dataset)))))

(defn get-num-batches
  [train-config]
  (count (:indexes train-config)))

(defrecord OnGPUBatchingSystem []
  PBatchingSystem
  (bs-setup [this train-config batch-type]
    (let [{:keys [dataset cv-dataset]} train-config
          batch-size (long (:batch-size train-config))
          _ (println "Uploading dataset")
          gpu-dataset (load-dataset-to-gpu dataset)
          gpu-cv-data (when cv-dataset
                        (load-dataset-to-gpu (take 1 cv-dataset)))
          train-cv-dataset (when cv-dataset
                             [(first gpu-cv-data) (second cv-dataset)])
          total-input-count (get-total-input-count (assoc train-config :gpu-dataset gpu-dataset))
          num-batches (/ total-input-count batch-size)
          [data labels] gpu-dataset
          n-input-vec (mapv (comp second cudnn/shape) data)
          [total-input-count _] (cudnn/shape (first data))
          ;;Labels is a vector of label matrixes so the outputs are a vector
          ;;of output sizes
          n-output-vec (if (= batch-type :training)
                         (mapv (comp second cudnn/shape) labels)
                         [])]
      (as-> train-config train-config
        (setup train-config batch-size total-input-count n-input-vec n-output-vec)
        (assoc train-config
               :gpu-dataset gpu-dataset
               :gpu-cv-dataset train-cv-dataset))))


  (bs-setup-epoch [bs train-config batch-type]
    (let [total-input-count (get-total-input-count train-config)
          batch-size (:batch-size train-config)]
     (if (or (= batch-type :testing)
             (= batch-type :running))
       (upload-sequential-indexes train-config total-input-count batch-size)
       (upload-randomized-indexes train-config total-input-count batch-size))))

  (bs-get-buffers [this train-config batch-idx batch-type]
    (let [gpu-data (if (or (= batch-type :training)
                           (= batch-type :running))
                     (first (:gpu-dataset train-config))
                     (first (:gpu-cv-dataset train-config)))
          input-buffers (get train-config data-keyword)
          output-buffers (get train-config label-keyword)
          train-config (load-batch-buffer train-config gpu-data
                                          batch-idx input-buffers)
          retval {:train-config train-config
                  :input-buffers input-buffers
                  :batch-indexes ((:indexes train-config) batch-idx)}]

      (if (= batch-type :training)
        (assoc retval
               :train-config (load-batch-buffer train-config (second (:gpu-dataset train-config))
                                                batch-idx output-buffers)
               :output-buffers output-buffers)
        retval)))
  (bs-get-cpu-labels [this train-config batch-type]
        (let [dataset (if (= batch-type :testing)
                    (:cv-dataset train-config)
                    (:dataset train-config))
          labels (second dataset)]
      (if (< (count (m/shape labels)) 3)
        [labels]
        labels)))

  (bs-has-cpu-labels? [this train-config batch-type]
    (bs-get-cpu-labels this train-config batch-type)))


(defn dataset-shape->array
  [shape batch-size]
  (if (number? shape)
    (cudnn/new-array [shape] batch-size)
    (let [{:keys [channel-count height width layout]} shape]
      (when-not (= layout ds/planar-image-layout)
        (throw (Exception. "Only planar image formats are supported at this time")))
      (cudnn/new-array batch-size channel-count height width))))

(defn array->batch-buffer
  [cudnn-buffer]
  (let [elem-count (cudnn/ecount cudnn-buffer)
        input-buffer-loader (cudnn/construct-ptr elem-count)
        input-transfer (double-array elem-count)]
    {:device-buffer cudnn-buffer
     :host-ptr input-buffer-loader
     :host-array input-transfer}))


(defn copy-items-to-transfer-buffer!
  [^doubles transfer-buffer item-seq]
  (first (reduce (fn [accum item]
                   (cuda/copy-to-double-array item (first accum) (second accum)))
                 [transfer-buffer 0]
                 item-seq)))


(defn upload-batch-data-to-buffer!
  [batch-buffer batch-data]
  (let [{:keys [device-buffer host-ptr host-array]} batch-buffer
        ^doubles host-array host-array
        elem-count (alength host-array)]
    (copy-items-to-transfer-buffer! host-array batch-data)
    (cudnn/copy-double-array->ptr host-array host-ptr)
    (cuda/mem-copy-host->device host-ptr (:ptr device-buffer) (* elem-count (cudnn/byte-size)))))


(defn create-batch-buffers
  [dataset indexes batch-size]
    (let [shapes (ds/shapes dataset)
          output-shapes (mapv shapes indexes)
          output-buffers (mapv #(array->batch-buffer (dataset-shape->array (:shape %) batch-size)) output-shapes)]
      output-buffers))


(defn get-or-create-batch-input-buffers
  [train-config input-indexes batch-size]
  (if-not (:input-buffers train-config)
    (assoc train-config :input-buffers (create-batch-buffers (:dataset train-config) input-indexes batch-size))
    train-config))


(defn get-or-create-batch-output-buffers
  [train-config output-indexes batch-size batch-type]
  (if (and (= batch-type :training)
           (not (:output-buffers train-config)))
    (assoc train-config :output-buffers (create-batch-buffers (:dataset train-config) output-indexes batch-size))
    train-config))


(defn upload-batch-data-to-buffers!
  [dataset-elements dataset-indexes buffer-seq]
  (let [element-seq (mapv (fn [dataset-index]
                            (mapv #(get % dataset-index) dataset-elements))
                          dataset-indexes)]
    (mapv (fn [elements buffer]
            (upload-batch-data-to-buffer! buffer elements)
            (:device-buffer buffer))
          element-seq
          buffer-seq)))


(defrecord DatasetBatchingSystem [input-dataset-indexes output-dataset-indexes batch-size]
  PBatchingSystem
  (bs-setup [this train-config batch-type]
    (let [dataset (:dataset train-config)]
      (-> train-config
          (get-or-create-batch-input-buffers input-dataset-indexes batch-size)
          (get-or-create-batch-output-buffers output-dataset-indexes batch-size batch-type))))


  (bs-setup-epoch [bs train-config batch-type]
    (let [dataset (:dataset train-config)
          indexes (ds/get-indexes dataset batch-type)
          indexes (if (= batch-type :training)
                    (shuffle indexes)
                    indexes)
          indexes (seq-to-partitioned-vec indexes batch-size)]
      (assoc train-config :indexes indexes)))

  (bs-get-buffers [this train-config batch-idx batch-type]
    (let [{:keys [input-buffers output-buffers dataset indexes]} train-config
          batch-indexes (indexes batch-idx)
          ds-elements (ds/get-elements dataset batch-indexes)]
      {:train-config train-config
       :input-buffers (upload-batch-data-to-buffers! ds-elements input-dataset-indexes input-buffers)
       :output-buffers (when (= batch-type :training)
                         (upload-batch-data-to-buffers! ds-elements output-dataset-indexes output-buffers))}))

  (bs-has-cpu-labels? [this train-config batch-type]
    (let [dataset (:dataset train-config)]
      (ds/has-indexes? dataset batch-type)))

  (bs-get-cpu-labels [this train-config batch-type]
    (let [dataset (:dataset train-config)
          indexes (ds/get-indexes dataset batch-type)
          ds-elements (map #(ds/get-element dataset %) indexes)]
      (mapv (fn [output-index]
              (mapv #(get % output-index) ds-elements))
            output-dataset-indexes))))
