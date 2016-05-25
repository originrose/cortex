(ns cortex-gpu.nn.description
  (:require [cortex.nn.description :as desc]
            [cortex.nn.caffe :as caffe]
            [cortex.nn.backends :as b]
            [cortex-gpu.nn.layers :as layers]
            [cortex-gpu.nn.cudnn :as cudnn]
            [clojure.core.matrix :as m]))


(defmulti create-module :type)

(defmethod create-module :input [desc] nil)

(defn to-cudnn-array
  [item]
  (when item
    (cudnn/array item)))

(defmethod create-module :linear
  [desc]
  (let [{:keys [input-size output-size weights bias l2-max-constraint]} desc]
    (layers/linear input-size output-size
                   :weights (to-cudnn-array weights)
                   :bias (to-cudnn-array bias)
                   :l2-max-constraint l2-max-constraint)))

(defmethod create-module :logistic
  [desc]
  (layers/sigmoid (:output-size desc)))

(defmethod create-module :relu
  [desc]
  (layers/relu (:output-size desc)))

(defmethod create-module :dropout
  [desc]
  (layers/dropout (:output-size desc) (:probability desc)))

(defmethod create-module :softmax
  [desc]
  (let [output-size (long (:output-size desc))
        n-channels (long (get desc :output-channels 1))
        n-input (quot output-size n-channels)]
   (layers/softmax n-input n-channels)))

(defmethod create-module :convolutional
  [{:keys [input-width input-height input-channels
           kernel-width kernel-height pad-x pad-y
           stride-x stride-y num-kernels
           weights bias l2-max-constraint] :as desc}]
  (layers/convolutional input-width input-height input-channels
                        kernel-width kernel-height pad-x pad-y
                        stride-x stride-y num-kernels
                        :weights (to-cudnn-array weights)
                        :bias (to-cudnn-array bias)
                        :l2-max-constraint l2-max-constraint))

(defmethod create-module :max-pooling
  [{:keys [input-width input-height input-channels
           kernel-width kernel-height pad-x pad-y
           stride-x stride-y]}]
  (layers/max-pooling input-width input-height input-channels
                      kernel-width kernel-height pad-x pad-y
                      stride-x stride-y))

(declare create-network)

(defmethod create-module :split
  [{:keys [branches input-size]}]
  (let [sub-networks (mapv create-network branches)]
    (layers/split sub-networks input-size)))

(defn create-network
  [built-descriptions]
  (let [modules (filterv identity (map create-module built-descriptions))]
    (layers/layer-list modules)))

(defn build-and-create-network
  [input-desc-seq]
  (create-network (desc/build-full-network-description input-desc-seq)))

(defn caffe->gpu-network
  [proto-model trained-model]
  (build-and-create-network (caffe/caffe->description proto-model trained-model)))

(extend-protocol desc/PNetworkToDescription
  cortex_gpu.nn.layers.Activation
  (layer->input [layer] (desc/input (:n-input layer)))
  (layer->description [layer]
    (let [layer-type (:activation-type layer)]
      (cond
        (= layer-type cudnn/activation-relu) (desc/relu)
        (= layer-type cudnn/activation-sigmoid) (desc/logistic))))
  cortex_gpu.nn.layers.Linear
  (layer->input [layer] (desc/input (layers/input-size layer)))
  (layer->description [layer] (desc/linear (layers/output-size layer)
                                           :weights (cudnn/to-core-matrix (:weights layer))
                                           :bias (cudnn/to-core-matrix (:bias layer))
                                           :l2-max-constraint (:l2-max-constraint layer)))

  cortex_gpu.nn.layers.Softmax
  (layer->input [layer] (desc/input (layers/input-size layer)))
  (layer->description [layer] (desc/softmax))

  cortex_gpu.nn.layers.Convolutional
  (layer->input [layer] (desc/conv-config->input (:conv-config layer)))
  (layer->description [layer]
    (desc/conv-config->description (:conv-config layer) :convolutional
                                   (cudnn/to-core-matrix (:weights layer))
                                   (cudnn/to-core-matrix (:bias layer))
                                   (:l2-max-constraint layer)))
  cortex_gpu.nn.layers.Pooling
  (layer->input [layer] (desc/conv-config->input (:conv-config layer)))
  (layer->description [layer]
    (desc/conv-config->description (:conv-config layer) :max-pooling))

  cortex_gpu.nn.layers.Dropout
  (layer->input [layer] (desc/input (layers/input-size layer)))
  (layer->description [layer] (desc/dropout (:probability layer)))

  ;;Be extremely careful about laziness in here because the underlying gpu resources
  ;;could be released before the lazy seq has been realized.
  cortex_gpu.nn.layers.LayerList
  (layer->input [layer] (desc/layer->input (first (:layers layer))))
  (layer->description [layer]
    (mapv desc/layer->description (:layers layer)))

  cortex_gpu.nn.layers.Split
  (layer->input [layer] (desc/input (layers/input-size layer)))
  (layer->description [layer] (desc/split (mapv desc/layer->description (:layers layer)))))


(defn network->description
  [network]
  (desc/network->description network))
