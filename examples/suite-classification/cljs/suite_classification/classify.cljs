(ns suite-classification.classify
  (:require [think.gate.core :as gate]
            [cortex.suite.classify :as classify]))


(defmethod gate/component "default"
  [& args]
  (apply classify/classify-component args))



(gate/start-frontend)
