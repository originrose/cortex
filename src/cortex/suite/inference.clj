(ns cortex.suite.inference
  (:require [cortex.dataset :as ds]
            [cortex.compute.nn.compute-execute :as ce]
            [cortex.nn.execute :as execute]
            [cortex.nn.traverse :as traverse]
            [cortex.nn.network :as network]
            [cortex.loss :as loss]
            [cortex.suite.train :as suite-train]
            [cortex.graph :as graph]
            [cortex.util :as util]))


(defn infer-n-observations
  "Given a network that has exactly one input and output, infer on these observations"
  [network observations observation-dataset-shape & {:keys [datatype batch-size force-cpu?]
                                                     :or {datatype :float
                                                          batch-size 1}}]
  (let [context (execute/create-context)
        ;;Creating an in-memory dataset with exactly 1 set of indexes causes it to use
        ;;that set of indexes without a shuffle or anything for all the different batch types.
        dataset (ds/->InMemoryDataset {:data {:data observations
                                              :shape observation-dataset-shape}}
                                      (vec (range (count observations))))
        ;;The loss terms appear like leaves to the graph confusion a few algorithms.
        network (traverse/remove-existing-loss-terms network)
        graph (network/network->graph network)
        roots  (graph/roots graph)
        leaves (graph/leaves graph)]
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
     :classification (get class-names (util/max-index result))}))
