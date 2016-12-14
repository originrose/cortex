(ns suite-classification.classify
  (:require-macros [cljs.core.async.macros :refer [go]])
  (:require [think.gate.core :as gate]
            [cljs.core.async :as async :refer [<!]]
            [reagent.core :refer [atom]]
            [think.gate.model :as model]
            [cortex.suite.classify :as classify]))


(defmethod gate/component "default"
  [& args]
  (apply classify/classify-component args))


(gate/start-frontend)
