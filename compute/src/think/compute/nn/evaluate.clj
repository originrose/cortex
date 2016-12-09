(ns think.compute.nn.evaluate
  (:require [think.compute.nn.train :as train]
            [think.compute.optimise :as opt]
            [think.compute.nn.layers :as layers]
            [think.compute.nn.description :as compute-desc]
            [think.resource.core :as resource]
            [cortex.dataset :as ds]))


(defn evaluate-softmax
  "Given a network, evaluate it against the answer set provided by the dataset."
  [net dataset input-labels & {:keys [output-index dataset-label-name batch-type]
                               :or {output-index 0
                                    dataset-label-name :labels
                                    batch-type :holdout}}]
  (let [run-results (nth (train/run net dataset input-labels) output-index)
        answer-seq (ds/get-data-sequence-from-dataset
                    dataset dataset-label-name
                    batch-type (layers/batch-size net))]
    (double (opt/evaluate-softmax run-results answer-seq))))


(defn evaluate-softmax-description
  [net-desc backend-fn dataset input-labels & {:keys [output-index dataset-label-name batch-type batch-size]
                                               :or {output-index 0
                                                    dataset-label-name :labels
                                                    batch-type :holdout
                                                    batch-size 10}}]
  (resource/with-resource-context
    (let [net (compute-desc/build-and-create-network net-desc (backend-fn) batch-size)]
      (evaluate-softmax net dataset input-labels :output-index output-index
                        :dataset-label-name dataset-label-name
                        :batch-type batch-type))))
