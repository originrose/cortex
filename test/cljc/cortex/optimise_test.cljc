(ns cortex.optimise-test
  (:require #?(:cljs
                [cljs.test :refer-macros [deftest is testing]]
                :clj
                [clojure.test :refer [deftest is testing]])
            [clojure.core.matrix :as m]
            [cortex.layers :as layers]
            [cortex.optimise :refer [adadelta-optimiser sgd-optimiser]]
            [cortex.core :refer [output forward backward parameter-count optimise stack-module calc-output]]))

;; simple optimiser testing function: try to optimse a transformation
(defn optimiser-test 
  ([m o]
    (optimiser-test m o nil))
  ([m o {:keys [iterations tolerance print-iterations]
         :as options}]
    (let [target [0.3 0.7]
          input [1 1]
          iterations (long (or iterations 200))
          tolerance (double (or tolerance 0.01))]
      (loop [i 0
             m m
             o o]
        (let [m (forward m input)
              m (backward m input (m/mul (m/sub (output m) target) 2.0))
              [o m] (optimise o m)
              result (calc-output m input)
              dist (m/length (m/sub result target))]
          (when print-iterations (println (str i " : error = " dist " : output = " result))) 
          (if (< i iterations)
            (recur (inc i) m o)
            (is (< dist tolerance))))))))

(deftest test-adadelta
  (let [m (layers/linear [[0.1 0.2] [0.3 0.4]] [0.1 0.1])
        o (adadelta-optimiser (parameter-count m))]
    (optimiser-test m o)))

(deftest test-sgd
  (let [m (layers/linear [[0.1 0.2] [0.3 0.4]] [0.1 0.1])
        o (sgd-optimiser (parameter-count m))]
    (optimiser-test m o)))

(deftest test-nn
  (let [m (stack-module
            [(layers/linear [[0.1 0.2] [0.3 0.4]] [0.1 0.1])
             (layers/logistic [2])])
        o (sgd-optimiser (parameter-count m))]
    (optimiser-test m o)))

(deftest test-relu-nn
  (let [m (stack-module
            [(layers/linear [[0.1 0.2] [0.3 0.4]] [0.1 0.1])
             (layers/relu [2])])
        o (sgd-optimiser (parameter-count m))]
    (optimiser-test m o)))

(deftest test-denoise
  (let [m (layers/denoising-autoencoder
            (stack-module
              [(layers/linear [[0.1 0.2] [0.3 0.4]] [0.1 0.1])
               (layers/logistic [2])])
            (stack-module
              [(layers/linear [[0.1 0.2] [0.3 0.4]] [0.1 0.1])
               (layers/logistic [2])]))
        o (sgd-optimiser (parameter-count m))]
    (optimiser-test m o {:iterations 300 :print-iterations false :tolerance 0.1})))

(defmacro pr-local
  [varname]
  `(println (str (name '~varname) ":") ~varname))
