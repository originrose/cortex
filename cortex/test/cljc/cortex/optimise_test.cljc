(ns cortex.optimise-test
  (:require
    #?(:cljs [cljs.test :refer-macros [deftest is testing]]
             :clj [clojure.test :refer [deftest is testing]])
    [clojure.core.matrix :as m]
    [clojure.core.matrix.random :as rand-matrix]
    [cortex.nn.layers :as layers]
    [cortex.optimise :as opt :refer [adadelta-optimiser sgd-optimiser newton-optimiser]]
    [cortex.nn.core :refer [output forward backward parameter-count optimise stack-module calc-output]]))

(defonce ^:dynamic *print-iterations* false)

(deftest test-clamp
  (testing "in-range values"
    (is (== 0 (opt/clamp-double 0 -1 1)))
    (is (== 0.1 (opt/clamp-double 0.1 -1 1)))
    (is (== -1 (opt/clamp-double -1 -1 1))))
  (testing "lower bound" 
    (is (== -1 (opt/clamp-double -1.1 -1 1)))
    (is (== -1 (opt/clamp-double Double/NEGATIVE_INFINITY -1 1))))
  (testing "upper bound"
    (is (== 1 (opt/clamp-double 2.0 -1 1)))
    (is (== 1 (opt/clamp-double Double/POSITIVE_INFINITY -1 1)))))

;; simple optimiser testing function: try to optimse a transformation
(defn optimiser-test
  ([m o]
    (optimiser-test m o nil))
  ([m o {:keys [iterations tolerance print-iterations noise]
         :as options}]
    (let [noise (or noise 0.0)
          base-target [0.3 0.7]
          input [1 1]
          iterations (long (or iterations 200))
          tolerance (double (or tolerance 0.01))]
      (loop [i 0
             m m
             o o]
        (let [m (forward m input)
              target (m/array :vectorz base-target) 
              _ (when (> noise 0) (m/add-scaled! target (rand-matrix/sample-normal [2]) noise))
              error-grad (m/mul (m/sub (output m) target) 2.0)
              m (backward m input error-grad)
              [o m] (optimise o m)
              result (calc-output m input)
              dist (m/length (m/sub result target))]
          (when print-iterations (println (str i " : error = " dist " : output = " result)))
          (if (< i iterations)
            (recur (inc i) m o)
            (is (< dist tolerance))))))))

(deftest test-adadelta
  (let [m (layers/linear [[0.1 0.2] [0.3 0.4]] [0.1 0.1])
        o (adadelta-optimiser)]
    (optimiser-test m o {:print-iterations *print-iterations*})
    ))

(deftest test-sgd
  (let [m (layers/linear [[0.1 0.2] [0.3 0.4]] [0.1 0.1])
        o (sgd-optimiser)]
    (optimiser-test m o {:print-iterations *print-iterations*})
    ))

(deftest test-nn
  (let [m (stack-module
            [(layers/linear [[0.1 0.2] [0.3 0.4]] [0.1 0.1])
             (layers/logistic [2])])
        o (adadelta-optimiser)]
    (optimiser-test m o {:print-iterations *print-iterations*})
    ))

(deftest test-nn-newton
  (let [m (stack-module
            [(layers/linear [[0.1 0.2] [0.3 0.4]] [0.1 0.1])
             (layers/logistic [2])])
        o (newton-optimiser)]
    (optimiser-test m o {:print-iterations *print-iterations*})
    ))

(defn relu-nn-test 
  "Tests with a small relu-based network for convergence with noisy target"
  [o]
  (let [m (stack-module
            [(layers/linear [[0.1 0.2] [0.3 0.4]] [0.1 0.1])
             (layers/relu [2])
             (layers/linear [[0.1 0.2] [0.3 0.4]] [0.1 0.1])
             (layers/relu [2])])
        o (sgd-optimiser)]
    (optimiser-test m o {:print-iterations *print-iterations*
                         :noise 0.01 ;; small amount of noise in optimisation
                         :tolerance 0.1})))

(deftest test-relu-nn
  (relu-nn-test (sgd-optimiser))
  (relu-nn-test (adadelta-optimiser))
  (relu-nn-test (newton-optimiser)))

(deftest test-mse-nulls
  (is (m/equals [1 2 3] (cortex.optimise/process-nulls [10 2 0] [1 nil 3])))
  (is (m/equals [1 2 3] (cortex.optimise/process-nulls [10 2 0] [1 nil 3]))))

(deftest test-denoise
  (let [m (layers/autoencoder
            (stack-module
              [(layers/linear [[0.1 0.2] [0.3 0.4]] [0.1 0.1])
               (layers/logistic [2])])
            (stack-module
              [(layers/linear [[0.1 0.2] [0.3 0.4]] [0.1 0.1])
               (layers/logistic [2])]))
        o (sgd-optimiser)]
    (optimiser-test m o {:iterations 300 :print-iterations false :tolerance 0.1})))

(defmacro pr-local
  [varname]
  `(println (str (name '~varname) ":") ~varname))
