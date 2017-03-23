(ns cortex.verify.nn.data
  (:require [cortex.datasets.mnist :as mnist]))


;; Data from: Dominick Salvator and Derrick Reagle
;; Shaum's Outline of Theory and Problems of Statistics and Economics
;; 2nd edition,  McGraw-Hill, 2002, pg 157

;; Predict corn yield from fertilizer and insecticide inputs
;; [corn, fertilizer, insecticide]

;; The text solves the model exactly using matrix techniques and determines
;; that corn = 31.98 + 0.65 * fertilizer + 1.11 * insecticides

(def CORN-DATA
  [[6  4]
   [10  4]
   [12  5]
   [14  7]
   [16  9]
   [18 12]
   [22 14]
   [24 20]
   [26 21]
   [32 24]])


(def CORN-LABELS
  [[40] [44] [46] [48] [52] [58] [60] [68] [74] [80]])


(def CORN-DATASET
  (mapv (fn [d l] {:data d :label l})
        CORN-DATA CORN-LABELS))


(defonce mnist-training-dataset* (future (mnist/training-dataset)))
(defonce mnist-test-dataset* (future (mnist/test-dataset)))
