(ns think.compute.verify.import
  (:require [think.compute.nn.description :as compute-desc]
            [think.compute.nn.backend :as backend]
            [cortex.nn.protocols :as cp]
            [clojure.core.matrix :as m]
            [think.resource.core :as resource]
            [cortex.nn.description :as desc]
            [think.compute.nn.cpu-backend :as cpu-backend]))


(defn verify-model
  ([backend-fn full-cortex-desc input layer-output-vec]
   (when-let [verification-failure (seq (desc/build-and-verify-trained-network full-cortex-desc))]
     (throw (Exception. (format "Description verification failed:\n %s"
                                (with-out-str (println verification-failure))))))

   (resource/with-resource-context
     (let [backend (backend-fn)
           network (compute-desc/build-and-create-network full-cortex-desc backend 1)
           ;;The assumption here is the network takes a single input and produces a single output
           net-input [(backend/array backend input)]
           network (cp/multi-calc network net-input)
           ;;Drop the input layer from the network
           network-working-layers (drop 1 (:layers network))
           ;;The assumption here is that we have a single linear network; this will not work for
           ;;branching networks.
           output-buffers (mapv (fn [previous-layer layer]
                                  [(when previous-layer
                                     (when-let [item-input (cp/output previous-layer)]
                                       (backend/to-double-array backend item-input)))
                                   (backend/to-double-array backend (cp/output layer))])
                                (concat [nil] network-working-layers) network-working-layers)]
       (vec (remove nil?
                    (map (fn [keras-output [net-input net-output] net-layer desc-layer]
                           (when keras-output
                             (when-not (m/equals keras-output net-output 0.001)
                               {:cortex-input net-input
                                :import-output keras-output
                                :cortex-output net-output
                                :description desc-layer})))
                         layer-output-vec output-buffers (:layers network) (drop 1 full-cortex-desc)))))))
  ([backend-fn {:keys [model input layer-outputs]}]
   (verify-model backend-fn model input layer-outputs))
  ([import-data]
   (verify-model #(cpu-backend/create-cpu-backend) import-data)))
