(ns cortex.verify.nn.import
  (:require [cortex.nn.network :as network]
            [cortex.nn.traverse :as traverse]
            [cortex.nn.execute :as execute]
            [cortex.nn.protocols :as cp]
            [cortex.graph :as graph]
            [think.resource.core :as resource]
            [clojure.pprint :as pprint]
            [clojure.core.matrix :as m]))



(defn verify-model
  ([context network layer-id->output]
   (when-not (contains? network :layer-graph)
     (throw (ex-info "Network appears to not be built (cortex.nn.network/build-network)"
                     {:network-keys (try (keys network)
                                         (catch Exception e []))})))
   (when-let [failures (seq (get network :verification-failures))]
     (throw (Exception. (format "Description verification failed:\n %s"
                                (with-out-str (clojure.pprint/pprint failures))))))
   (resource/with-resource-context
     (let [roots (graph/roots (network/network->graph network))
           leaves (graph/leaves (network/network->graph network))
           input-bindings {(first roots) :data}
           input (get layer-id->output (first roots))
           output-bindings (->> leaves
                                (map #(vector % {}))
                                (into {}))
           network
           (as-> (traverse/auto-bind-io network) network
             (traverse/network->inference-traversal network {:data (m/ecount input)})
             (assoc network :batch-size 1)
             (cp/bind-to-network context network {})
             (cp/traverse context network {:data input} :inference)
             ;;save gradients at this point implies save io buffers
             (cp/save-to-network context network {:save-gradients? true}))
           traversal (get-in network [:traversal :forward])
           io-buffers (get-in network [:traversal :buffers])]
       (->> traversal
            (map (fn [{:keys [incoming id outgoing]}]
                   (when-let [import-output (get layer-id->output id)]
                     (println "verifying output for layer:" id)
                     (let [import-output (m/as-vector import-output)
                           buffer-data (m/as-vector
                                        (get-in io-buffers [(first outgoing) :buffer]))]
                       (when-not (m/equals import-output buffer-data 1e-3)
                         {:cortex-input (m/as-vector (get-in io-buffers [(first incoming) :buffer]))
                          :import-output import-output
                          :cortex-output buffer-data
                          :layer (get-in network [:layer-graph :id->node-map id])})))))
            (remove nil?)
            vec))))
  ([context {:keys [model layer-id->output]}]
   (verify-model context model layer-id->output)))
