(ns cortex.suite.inference
  (:require [cortex.dataset :as ds]
            [think.compute.nn.cpu-backend :as cpu-backend]
            [think.compute.nn.compute-execute :as ce]
            [cortex.nn.execute :as execute]
            [cortex.nn.traverse :as traverse]
            [cortex.nn.network :as network]
            [cortex.loss :as loss]))


(defn infer-n-observations
  "Given a network that has exactly one input and output, infer on these observations"
  [network observations observation-dataset-shape & {:keys [datatype batch-size force-cpu?]
                                                     :or {datatype :float
                                                          batch-size 1}}]
  (let [backend-fn (fn []
                     (or (when-not force-cpu?
                           (try
                             (require 'think.compute.nn.cuda-backend)
                             ((resolve 'think.compute.nn.cuda-backend/create-backend) datatype)
                             (catch Exception e
                               (println (format "Failed to create cuda backend (%s); will use cpu backend"
                                                e))
                               nil)))
                         (cpu-backend/create-cpu-backend datatype)))
        context (ce/create-context backend-fn)
        ;;Creating an in-memory dataset with exactly 1 set of indexes causes it to use
        ;;that set of indexes without a shuffle or anything for all the different batch types.
        dataset (ds/->InMemoryDataset {:data {:data observations
                                              :shape observation-dataset-shape}}
                                      (vec (range (count observations))))
        [roots leaves] (network/edges->roots-and-leaves (network/network->edges network))]
    (when-not (= 1 (count roots))
      (throw (ex-info "Network must have exactly 1 root for infer-n-observations"
                      {:roots roots})))
    (when-not (= 1 (count leaves))
      (throw (ex-info "Network must have exactly 1 leaf for infer-n-observations"
                      {:leaves leaves})))
    (when-not (= 0 (rem (count observations)
                        batch-size))
      (throw (ex-info "Batch size does not evenly divide into the number of observations"
                      {:batch-size batch-size
                       :observation-count (count observations)})))
    (as-> (traverse/auto-bind-io network) network-or-seq
      (execute/infer-columns context network-or-seq dataset [] [] :batch-size batch-size)
      (get network-or-seq (first leaves)))))


(defn classify-one-observation
  "observation-dataset-shape is something like (ds/create-image-shape num-channels img-width img-height)"
  [network observation observation-dataset-shape class-names]
  (let [results (infer-n-observations network [observation] observation-dataset-shape
                                      :batch-size 1
                                      :force-cpu? true)
        result (vec (first results))]
    {:probability-map (into {} (map vec (partition 2 (interleave class-names result))))
     :classification (get class-names (loss/max-index result))}))
