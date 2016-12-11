(ns think.compute.nn.description
  (:require [think.compute.nn.layers :as layers]
            [cortex.nn.description :as desc]
            [think.compute.nn.backend :as nn-backend]
            [cortex.nn.protocols :as cp]))


(defmulti create-module (fn [desc backend] (:type desc)))

(defmethod create-module :input [desc backend] (layers/->InputLayer backend desc))

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
  (let [n-input (long (:output-size desc))
        n-channels (long (get desc :output-channels 1))]
    (layers/softmax backend n-input :channels n-channels)))


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


(defn save-desc
  "Save the description with the layer so that we save any extra information
in the description that we don't know about.  At this time it seems best to think
of the entire system as a way of adding annotations to descriptions and in this light
it seems clear that if there is any other information on the description it should
be preserved."
  [desc layer]
  (when layer
    (let [layer (if-let [atten (get desc :learning-attenuation)]
                  (assoc layer :learning-attenuation atten)
                  layer)]
      (assoc layer :source-description desc))))


(defn description->module
  [backend desc]
  (->> (create-module desc backend)
       (save-desc desc)))


(defn create-network
  [built-descriptions backend batch-size]
  (let [modules (->> (map (partial description->module backend) built-descriptions)
                     (filterv identity))]
    (cp/setup (layers/layer-list modules) batch-size)))


(defn build-and-create-network
  [input-desc-seq backend batch-size]
  (create-network (desc/build-full-network-description input-desc-seq) backend batch-size))


(defn merge-with-original-desc
  [new-desc-seq layer-desc]
  (if (vector? new-desc-seq)
    (update-in new-desc-seq [0] #(merge layer-desc %))
    (merge layer-desc new-desc-seq)))


(defn get-layer-input
  [layer]
  (get layer :input-description))


(defn get-layer-source
  [layer]
  (get layer :source-description))


(defn merge-with-layer-input
  [new-desc-seq layer]
  (merge-with-original-desc new-desc-seq (get-layer-input layer)))


(defn merge-with-layer-source
  [new-desc-seq layer]
  (merge-with-original-desc new-desc-seq (get-layer-source layer)))


(defn compute-layer->description
  [layer]
  (-> (layers/->description layer)
      (merge-with-layer-source layer)))


(extend-protocol desc/PNetworkToDescription
  think.compute.nn.layers.InputLayer
  (layer->input [layer] (layers/->input layer))
  (layer->description [layer])

  think.compute.nn.layers.SimpleLayer
  (layer->input [layer] (layers/->input layer))
  (layer->description [layer] (compute-layer->description layer))

  think.compute.nn.layers.Linear
  (layer->input [layer] (layers/->input layer))
  (layer->description [layer] (compute-layer->description layer))

  think.compute.nn.layers.Convolutional
  (layer->input [layer] (layers/->input layer))
  (layer->description [layer] (compute-layer->description layer))

  think.compute.nn.layers.Pooling
  (layer->input [layer] (layers/->input layer))
  (layer->description [layer] (compute-layer->description layer))

  think.compute.nn.layers.Dropout
  (layer->input [layer] (layers/->input layer))
  (layer->description [layer] (compute-layer->description layer))

  ;;Be extremely careful about laziness in here because the underlying gpu resources
  ;;could be released before the lazy seq has been realized.
  think.compute.nn.layers.LayerList
  (layer->input [layer] (desc/layer->input (first (:layers layer))))
  (layer->description [layer] (->>
                               (mapv desc/layer->description (:layers layer))
                               ;;Filter out input descriptor
                               (remove nil?)
                               vec))

  think.compute.nn.layers.Split
  (layer->input [layer] (layers/->input layer))
  (layer->description [layer] (compute-layer->description layer))

  think.compute.nn.layers.BatchNormalization
  (layer->input [layer] (layers/->input layer))
  (layer->description [layer] (compute-layer->description layer)))
