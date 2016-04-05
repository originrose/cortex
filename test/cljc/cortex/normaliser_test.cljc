(ns cortex.normaliser-test
  (:require #?(:cljs
                [cljs.test :refer-macros [deftest is testing]]
               :clj
                [clojure.test :refer [deftest is testing]])
            [clojure.core.matrix :as m]
            [clojure.core.matrix.stats :as stats]
            [cortex.layers :as layers]
            [cortex.optimise :refer [adadelta-optimiser]]
            [cortex.core :refer [output forward backward parameter-count optimise]]))

(def DATA 
  [[-1 10  0]
   [-1 -10 1]
   [-1 10  2]
   [-1 -10 0]
   [-1 0   2]])
;; mean [-1.0 0.0 1.0]
;; sd   [0.0 10.0 1.0]

;; normaliser testing function
(defn normaliser-test [m o]
  (loop [i 0
           m m
           o o]
      (let [input (nth DATA (mod i (count DATA))) 
            m (forward m input)
            m (backward m input [0 0 0])
            [o m] (optimise o m)
            dist (m/length (m/sub (output m) [1 1 1]))]
        ;; (if (== 0 (mod i 20)) (println (str i " : " (output m))))
      (if (< i 1000) 
          (recur (inc i) m o)
          m))))

(deftest test-nn
  (let [m (layers/normaliser [3] {:learn-rate 0.03 :normaliser-factor 0.01})
        o (adadelta-optimiser (parameter-count m))
        m (normaliser-test m o)] 
    (is (< (m/emax (m/abs (m/sub (:mean m) [-1.0 0.0 1.0]))) 0.2))
    ))

