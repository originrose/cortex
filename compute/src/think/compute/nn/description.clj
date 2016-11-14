(ns think.compute.nn.description
  (:require [think.compute.nn.layers :as layers]
            [cortex.nn.description :as desc]
            [think.compute.nn.backend :as nn-backend]
            [cortex.nn.protocols :as cp]))


(defmulti create-module (fn [desc backend] (:type desc)))

(defmethod create-module :input [desc backend] nil)

(defn to-network-array
  [backend item]
  (when item
    (nn-backend/array backend item)))


(defmethod create-module :linear
  [desc backend]
  (let [{:keys [input-size output-size weights bias l2-max-constraint]} desc]
    (layers/linear backend input-size output-size
                   :weights (to-network-array backend weights)
                   :bias (to-network-array backend bias)
                   :l2-max-constraint l2-max-constraint)))


(defmethod create-module :logistic
  [desc backend]
  (layers/sigmoid backend (:output-size desc)))


(defmethod create-module :relu
  [desc backend]
  (layers/relu backend (:output-size desc)))


(defmethod create-module :dropout
  [desc backend]
  (if (= (:distribution desc) :bernoulli)
    (layers/bernoulli-dropout backend (:output-size desc) (:probability desc))
    (layers/gaussian-dropout backend (:output-size desc) (- 1.0 (:probability desc)))))


(defmethod create-module :softmax
  [desc backend]
  (let [output-size (long (:output-size desc))
        n-channels (long (get desc :output-channels 1))
        n-input (quot output-size n-channels)]
    (when-not (= n-channels 1)
      (throw (Exception. "Multi-channel softmax not supported at this time.")))
   (layers/softmax backend n-input)))

(defmethod create-module :convolutional
  [{:keys [input-width input-height input-channels
           kernel-width kernel-height pad-x pad-y
           stride-x stride-y num-kernels
           weights bias l2-max-constraint] :as desc} backend]
  (layers/convolutional backend
                        input-width input-height input-channels
                        kernel-width kernel-height pad-x pad-y
                        stride-x stride-y num-kernels
                        :weights (to-network-array backend weights)
                        :bias (to-network-array backend bias)
                        :l2-max-constraint l2-max-constraint))

(defmethod create-module :max-pooling
  [{:keys [input-width input-height input-channels
           kernel-width kernel-height pad-x pad-y
           stride-x stride-y]} backend]
  (layers/max-pooling backend
                      input-width input-height input-channels
                      kernel-width kernel-height pad-x pad-y
                      stride-x stride-y))

(defmethod create-module :batch-normalization
  [{:keys [output-size average-factor epsilon means variances scale bias]} backend]
  (layers/batch-normalization backend output-size average-factor
                              :scale (to-network-array backend scale)
                              :bias (to-network-array backend bias)
                              :means (to-network-array backend means)
                              :variances (to-network-array backend variances)
                              :epsilon epsilon))

(defmethod create-module :local-response-normalization
  [{:keys [n k alpha beta output-width output-height output-channels] :as desc} backend]
  (layers/local-response-normalization backend output-width output-height output-channels
                                       :k k :n n :alpha alpha :beta beta))

(declare create-network)

(defmethod create-module :split
  [{:keys [branches input-size]} backend]
  (let [sub-networks (mapv create-network branches)]
    (layers/split backend sub-networks input-size)))

(defn create-network
  [built-descriptions backend batch-size]
  (let [modules (filterv identity (map #(create-module % backend) built-descriptions))]
    (cp/setup (layers/layer-list modules) batch-size)))

(defn build-and-create-network
  [input-desc-seq backend batch-size]
  (create-network (desc/build-full-network-description input-desc-seq) backend batch-size))

(defn convert-layer-type
  [compute-type]
  (if (= compute-type :sigmoid)
    :logistic
    compute-type))

(defmulti simple-layer->input (fn [layer]
                                (get-in layer [:layer-impl-desc
                                               :layer-type])))

(defmethod simple-layer->input :default
  [layer]
  (desc/input (:n-input layer)))


(defmethod simple-layer->input :local-response-normalization
  [layer]
  (let [{:keys [width height n-channels]} (:layer-impl-desc layer)]
    (desc/input width height n-channels )))


(defmulti simple-layer->description (fn [layer]
                                      (get-in layer [:layer-impl-desc
                                                     :layer-type])))


(defmethod simple-layer->description :default
  [layer]
  {:type (convert-layer-type (get-in layer [:layer-impl-desc
                                            :layer-type]))})

(defmethod simple-layer->description :local-response-normalization
  [layer]
  (let [{:keys [k n alpha beta]} (:layer-impl-desc layer)]
    (desc/local-response-normalization
     :k k :n n :alpha alpha :beta beta)))


(extend-protocol desc/PNetworkToDescription
  think.compute.nn.layers.SimpleLayer
  (layer->input [layer] (simple-layer->input layer))
  (layer->description [layer] (simple-layer->description layer))

  think.compute.nn.layers.Linear
  (layer->input [layer] (desc/input (cp/input-size layer)))
  (layer->description [layer] (desc/linear (cp/output-size layer)
                                           :weights (nn-backend/to-core-matrix
                                                     (layers/get-backend layer)
                                                     (:weights layer))
                                           :bias (nn-backend/to-core-matrix
                                                  (layers/get-backend layer)
                                                  (:bias layer))
                                           :l2-max-constraint (:l2-max-constraint layer)))

  think.compute.nn.layers.Convolutional
  (layer->input [layer] (desc/conv-config->input (:conv-config layer)))
  (layer->description [layer]
    (desc/conv-config->description (:conv-config layer) :convolutional
                                   (nn-backend/to-core-matrix (layers/get-backend layer)
                                                           (:weights layer))
                                   (nn-backend/to-core-matrix (layers/get-backend layer)
                                                           (:bias layer))
                                   (:l2-max-constraint layer)))

  think.compute.nn.layers.Pooling
  (layer->input [layer] (desc/conv-config->input (:conv-config layer)))
  (layer->description [layer]
    (desc/conv-config->description (:conv-config layer) :max-pooling))

  think.compute.nn.layers.Dropout
  (layer->input [layer] (desc/input (cp/input-size layer)))
  (layer->description [layer]
    (if (= (get-in layer [:dropout-options :distribution]) :bernoulli)
      (desc/dropout (get-in layer [:dropout-options :probability])
                    :distribution :bernoulli)
      (desc/dropout (- 1.0 (get-in layer [:dropout-options :variance]))
                    :distribution :gaussian)))

  ;;Be extremely careful about laziness in here because the underlying gpu resources
  ;;could be released before the lazy seq has been realized.
  think.compute.nn.layers.LayerList
  (layer->input [layer] (desc/layer->input (first (:layers layer))))
  (layer->description [layer]
    (mapv desc/layer->description (:layers layer)))

  think.compute.nn.layers.Split
  (layer->input [layer] (desc/input (cp/input-size layer)))
  (layer->description [layer] (desc/split (mapv desc/layer->description (:layers layer))))

  think.compute.nn.layers.BatchNormalization
  (layer->input [layer] (desc/input (cp/input-size layer)))
  (layer->description [layer]
    (let [core-mat (fn [data] (nn-backend/to-core-matrix (layers/get-backend layer) data))]
      (merge (first
              (desc/batch-normalization (:average-factor layer)
                                        :epsilon (:epsilon layer)))
             {:scale (core-mat (:scale layer))
              :bias (core-mat (:bias layer))
              :means (core-mat (:running-means layer))
              :variances (core-mat (:running-variances layer))}))))
