(ns think.compute.nn.evaluate
  (:require [think.compute.nn.train :as train]
            [think.compute.optimise :as opt]
            [cortex.dataset :as ds]))


(defn evaluate-softmax
  "Given a network, evaluate it against the answer set provided by the dataset."
  [net dataset input-labels & {:keys [output-index dataset-label-index]
                               :or {output-index 0
                                    dataset-label-index 1}}]
  (let [run-results (nth (train/run net dataset input-labels) output-index)
        answer-seq (first (ds/get-elements dataset (ds/get-indexes dataset :running) [dataset-label-index]))]
    (double (opt/evaluate-softmax run-results answer-seq))))
