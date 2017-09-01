(ns cortex.verify.nn.import
  (:require [cortex.nn.network :as network]
            [cortex.nn.traverse :as traverse]
            [cortex.nn.execute :as execute]
            [cortex.nn.compute-binding :as compute-binding]
            [cortex.graph :as graph]
            [think.resource.core :as resource]
            [clojure.pprint :as pprint]
            [clojure.core.matrix :as m]))



(defn verify-model
  ([context network layer-id->output]
   (when-not (contains? network :compute-graph)
     (throw (ex-info "Network appears to not be built (cortex.nn.network/linear-network)"
                     {:network-keys (try (keys network)
                                         (catch Exception e []))})))
   (when-let [failures (seq (get network :verification-failures))]
     (throw (Exception. (format "Description verification failed:\n %s"
                                (with-out-str (clojure.pprint/pprint failures))))))
   (execute/with-compute-context context
     (let [roots (graph/roots (network/network->graph network))
           leaves (graph/leaves (network/network->graph network))
           input (get layer-id->output (first roots))
           batch-size 1
           network
           (as-> network network
             ;;Disable pooling as that will definitely break the buffer-by-buffer comparison
             (compute-binding/bind-context-to-network network
                                                      (execute/current-backend)
                                                      batch-size (traverse/training-traversal network)
                                                      {:disable-pooling? true})
               (compute-binding/traverse context network {(first roots) input} :inference)
               ;;save gradients at this point implies save io buffers
               (compute-binding/save-to-network context network {:save-gradients? true}))
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
                         {:cortex-input  (m/as-vector (get-in io-buffers [(first incoming) :buffer]))
                          :import-output import-output
                          :cortex-output buffer-data
                          :layer         (get-in network [:compute-graph :nodes id])})))))
            (remove nil?)
            vec))))
  ([context {:keys [model layer-id->output]}]
   (verify-model context model layer-id->output)))
