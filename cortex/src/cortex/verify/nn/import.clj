(ns cortex.verify.nn.import
  (:require [cortex.nn.network :as network]
            [cortex.nn.traverse :as traverse]
            [cortex.nn.execute :as execute]
            [cortex.nn.protocols :as cp]
            [think.resource.core :as resource]
            [clojure.pprint]
            [clojure.core.matrix :as m]))



(defn verify-model
  ([context network input layer-output-vec]
   (when-not (contains? network :layer-graph)
     (throw (ex-info "Network appears to not be built (cortex.nn.network/build-network)"
                     {:network-keys (try (keys network)
                                         (catch Exception e []))})))
   (when-let [failures (seq (get network :verification-failures))]
     (throw (Exception. (format "Description verification failed:\n %s"
                                (with-out-str (clojure.pprint/pprint failures))))))
   (resource/with-resource-context
     (let [[roots leaves] (network/edges->roots-and-leaves (get-in network [:layer-graph :edges]))
           input-bindings {(first roots) :data}
           output-bindings (->> leaves
                                (map #(vector % {}))
                                (into {}))
           network
           (as-> (traverse/auto-bind-io network) network
             (traverse/network->inference-traversal network)
             (assoc network :batch-size 1)
             (cp/bind-to-network context network {})
             (cp/traverse context network {:data input} :inference)
             ;;save gradients at this point implies save io buffers
             (cp/save-to-network context network {:save-gradients? true}))
           traversal (get-in network [:traversal :forward])
           io-buffers (get-in network [:traversal :buffers])]
       (->> layer-output-vec
            (map (fn [{:keys [incoming id outgoing]}  keras-output]
                   (when keras-output
                     (let [keras-output (m/as-vector keras-output)
                           buffer-data (m/as-vector
                                        (get-in io-buffers [(first outgoing) :buffer]))]
                       (when-not (m/equals keras-output buffer-data 1e-3)
                         {:cortex-input (m/as-vector (get-in io-buffers [(first incoming) :buffer]))
                          :import-output keras-output
                          :cortex-output buffer-data
                          :layer (get-in network [:layer-graph :id->node-map id])}))))
                 traversal)
            (remove nil?)
            vec))))
  ([context {:keys [model input layer-outputs]}]
   (verify-model context model input layer-outputs)))
