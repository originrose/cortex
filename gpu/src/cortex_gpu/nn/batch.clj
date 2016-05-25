(ns cortex-gpu.nn.batch
  (:require [cortex-gpu.nn.cudnn :as cudnn]
            [cortex-gpu.nn.layers :as layers]
            [clojure.core.matrix :as m]
            [cortex-gpu.cuda :as cuda])
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
      [(cudnn/array data)
       (cudnn/vec-of-matrixes labels)]
      [(cudnn/array data)])))


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


(defn upload-randomized-indexes
  [network ^long dataset-count]
  (upload-indexes network (int-array (shuffle (range dataset-count)))))


(defn upload-sequential-indexes
  [network ^long dataset-count]
  (upload-indexes network (int-array (range dataset-count))))


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
  [network batch-size total-input-count n-input n-output]
  (-> network
      (allocate-batch-index-buffer total-input-count)
      (setup-batch-buffer batch-size n-input data-keyword)
      (setup-batch-buffer batch-size n-output label-keyword)))
