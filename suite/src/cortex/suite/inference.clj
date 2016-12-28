(ns cortex.suite.inference
  (:require [cortex.dataset :as ds]
            [think.compute.nn.cpu-backend :as cpu-backend]
            [think.compute.nn.compute-execute :as ce]
            [think.resource.core :as resource]
            [cortex.optimise :as cortex-opt]))


(defn infer-n-observations
  "Given a network that has exactly one input and output, infer on these observations"
  [network observations observation-dataset-shape datatype & {:keys [cuda-backend?]}]
  (resource/with-resource-context
    (let [context ()
          backend (when-not backend
                    (cpu-backend/create-cpu-backend datatype))
          network (compute-desc/build-and-create-network network-description backend 1)
          n-observations (count observations)]
      (first (train/run
               network
               (ds/->InMemoryDataset {:data {:data observations
                                             :shape observation-dataset-shape}}
                                     (vec (range n-observations)))
               [:data])))))


(defn classify-one-observation
  "data-shape is
(ds/create-image-shape num-channels img-width img-height)"
  [network-description observation observation-dataset-shape
   datatype class-names & {:keys [cuda-backend?]}]
  (let [results (infer-n-observations network-description [observation] observation-dataset-shape
                                      datatype :cuda-backend? cuda-backend?)
        result (vec (first results))]
    {:probability-map (into {} (map vec (partition 2 (interleave class-names result))))
     :classification (get class-names (opt/max-index result))}))
