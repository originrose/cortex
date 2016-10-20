(ns think.cortex.keras.verify
  (:require [clojure.test :refer :all]
            [think.cortex.keras.core :as keras]
            [think.compute.nn.cpu-backend :as cpu-back]
            [cortex.nn.description :as desc]
            [cortex.nn.protocols :as cp]
            [think.compute.nn.backend :as backend]
            [clojure.core.matrix :as m]
            [think.compute.nn.description :as compute-desc]
            [think.resource.core :as resource]
            [think.hdf5.core :as hdf5]))

(defn- verify-model-on-backend
  [backend full-cortex-desc input layer-output-vec]
  (let [backend (cpu-back/create-cpu-backend)
        network (compute-desc/build-and-create-network full-cortex-desc backend 1)
        ;;The assumption here is the network takes a single input and produces a single output
        net-input [(backend/array backend input)]
        network (cp/multi-calc network net-input)
        ;;The assumption here is that we have a single linear network; this will not work for
        ;;branching networks.
        output-buffers (mapv (comp #(backend/to-double-array backend %) cp/output) (:layers network))]
    (vec (remove nil?
                 (map (fn [keras-output net-output net-layer desc-layer]
                        (when keras-output
                          (when-not (m/equals keras-output net-output 0.001)
                            {:layer-id (:id desc-layer)
                             :keras-output keras-output
                             :network-output net-output
                             :layer-type (:type desc-layer)})))
                      layer-output-vec output-buffers (:layers network) (drop 1 full-cortex-desc))))))


(defn verify-model
  ([full-cortex-desc input layer-output-vec {:keys [cpu gpu]
                                             :or {cpu true gpu false} :as opts}]
   (when-let [verification-failure (seq (desc/build-and-verify-trained-network full-cortex-desc))]
     (throw (Exception. (format "Description verification failed:\n %s"
                                (with-out-str (clojure.pprint/pprint verification-failure))))))
   (comment
    :gpu
    (when gpu
      (resource/with-resource-context
        (verify-model-on-backend (cuda-backend/create-backend :double) full-cortex-desc input layer-output-vec))))
   {:cpu
    (when cpu
      (resource/with-resource-context
        (verify-model-on-backend (cpu-back/create-cpu-backend) full-cortex-desc input layer-output-vec)))
    })
  ([{:keys [model input layer-outputs]} opts]
   (verify-model model input layer-outputs opts)))
