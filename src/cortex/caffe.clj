(ns cortex.caffe ^{:author "thinktopic"
                   :doc "Caffe integration of cortex.  Caffe expects and produces planar data anywhere it has multiple channels
while cortex works with interleaved data.
https://github.com/Yangqing/caffe/wiki/Convolution-in-Caffe:-a-memo
We fix this when creating objects via interleaving the weights on any object that was expecting planar input data.  Then we can
run normally with no further changes assuming the layers are implemented correctly.  This weight interleaving
is accomplished in description.clj"}
  (:require [clojure.java.io :as io]
            [cortex.serialization :as s]
            [cortex.description :as desc]
            [clojure.core.matrix :as m]
            [cortex.backends :as b])
  (:import [caffe Caffe Caffe$NetParameter]
           [com.google.protobuf CodedInputStream TextFormat]))


(defn create-coded-input-stream
  ^CodedInputStream [path-or-whatever]
  (let [input-stream (io/input-stream path-or-whatever)
        coded-stream (CodedInputStream/newInstance input-stream)]
    ;;We deal in big stuff here
    (.setSizeLimit coded-stream Integer/MAX_VALUE)
    coded-stream))


(defn load-binary-caffe-file
  "Load a caffe model from binary file (.caffemodel)"
  ^Caffe$NetParameter [path-or-whatever]
  (Caffe$NetParameter/parseFrom (create-coded-input-stream path-or-whatever)))

(defn load-text-caffe-file
  "Load a net parameter from a text file (.prototxt)"
  ^Caffe$NetParameter [path-or-whatever]
  (let [reader (io/reader path-or-whatever)
        builder (Caffe$NetParameter/newBuilder)]
    (TextFormat/merge reader builder)
    (.build builder)))

(defmulti read-layer (fn [layer-param] (.getType layer-param)))

(defn first-with-default
  [item-list default]
  (if-let [first-item (first item-list)]
    first-item
    0))

(defn pooling-param-to-map
  [pool-param]
  (let [stride (.getStride pool-param)
        pad (.getPad pool-param)
        kern-size (.getKernelSize pool-param)]
    { :kernel-width kern-size :kernel-height kern-size
     :pad-x pad :pad-y pad :stride-x stride :stride-y stride}))

(defn conv-param-to-map
  [conv-param]
  (let [stride (first-with-default (.getStrideList conv-param) 1)
        pad (first-with-default (.getPadList conv-param) 0)
        kern-size (first-with-default (.getKernelSizeList conv-param) 1)]
    { :kernel-width kern-size :kernel-height kern-size
     :pad-x pad :pad-y pad :stride-x stride :stride-y stride}))


(defmethod read-layer "Convolution"
  [layer-param]
  (let [blobs (.getBlobsList layer-param)
        weights (.getDataList (first blobs))
        weight-shape (into [] (.getDimList (.getShape (first blobs))))
        [n-kernels _ kern-width kern-height] weight-shape
        bias (m/reshape
              (b/array (seq (.getDataList (second blobs))))
              [1 n-kernels])
        n-channels (quot (count weights) (* kern-width kern-height n-kernels))
        kern-stride (* kern-width kern-height n-channels)
        weights (m/reshape (b/array (seq weights))
                           [n-kernels (* kern-width kern-height n-channels)])
        param-map (conv-param-to-map (.getConvolutionParam layer-param))]
    (assoc param-map
           :type :convolutional
           :num-kernels n-kernels
           :weights weights
           :bias bias
           :output-data-format :planar)))

(defmethod read-layer "Pooling"
  [layer-param]
  (assoc (pooling-param-to-map (.getPoolingParam layer-param))
         :type :max-pooling
         :input-data-format :planar
         :output-data-format :planar))

(defmethod read-layer "InnerProduct"
  [layer-param]
  (let [blobs (.getBlobsList layer-param)
        weights (.getDataList (first blobs))
        [n-output n-input] (into [] (.getDimList (.getShape (first blobs))))
        bias (b/array  (seq (.getDataList (second blobs))))
        weights (m/reshape (b/array (seq weights)) [n-output n-input])
        ]
    (assoc (first (desc/linear n-output))
           :weights weights
           :bias bias)))

(defmethod read-layer "ReLU"
  [layer-param]
  {:type :relu })

(defmethod read-layer "SoftmaxWithLoss"
  [layer-param]
  {:type :softmax})

(defmethod read-layer "Softmax"
  [layer-param]
  {:type :softmax})


(defmethod read-layer :default
  [layer-param]
  (println "Found unknown layer type: " (.getType layer-param)))


(defn model-to-input
  [model]
  (let [input-shape-data (first (.getInputShapeList model))
        dims (into [] (.getDimList input-shape-data))
        [input-batch-size n-channels width height] dims]
    (assoc (first (desc/input width height n-channels)) :output-data-format :planar)))


(defn caffe->description
  [proto-model trained-model]
  (let [all-layers (rest (.getLayerList trained-model))
        input-layer (model-to-input proto-model)]
    (concat [input-layer] (map #(read-layer %) all-layers))))


(defn instantiate-model
  "Given caffe models loaded with the above functions (load-X-caffe-file)
instantiate a cortex network"
  [proto-model trained-model]
  (desc/build-and-create-network (caffe->description proto-model trained-model)))
