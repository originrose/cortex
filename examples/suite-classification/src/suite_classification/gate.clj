(ns suite-classification.gate
  (:require [think.gate.core :as gate]
            [suite-classification.core :as core]
            [cortex.suite.classification :as classification]
            [clojure.core.matrix :as m]
            [cortex.dataset :as ds]
            [clojure.java.io :as io]))


(defn train-forever
  []
  (let [gate-message
        (gate/open #'classification/routing-map :port 8091)]
    (println gate-message))
  (classification/web-train-forever (core/create-dataset) core/mnist-observation->image (core/create-basic-mnist-description)))
