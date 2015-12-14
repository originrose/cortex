(ns cortex.test-utils
  (:require [clojure.core.matrix :as m])
  (:use [clojure.test]))

(def DEFAULT-TOLERANCE 0.001)
(def DEFAULT-MAX-TESTS 100)

(defn converges?
  "Tests if a sequence of array values converges to a target value, with a given tolerance. 
   Returns nil if convergence does not happen, the success value from the sequence if it does."
  ([sequence target]
    (converges? sequence target nil))
  ([sequence target {:keys [tolerance max-tests test-fn hits-needed] :as options}]
    (let [tolerance (or tolerance DEFAULT-TOLERANCE)
          max-tests (long (or max-tests DEFAULT-MAX-TESTS))
          test-fn (or test-fn identity)
          hits-needed (long (or hits-needed 1))]
      (loop [i 0
             hits 0
             sequence (seq sequence)]
        (when (< i max-tests)
          (if-let [v (first sequence)]
             (if (m/equals target (test-fn v) tolerance) ;; equals with tolerance
               (if (>= (inc hits) hits-needed) 
                 v
                 (recur (inc i) (inc hits) (next sequence)))
               (recur (inc i) 0 (next sequence)))))))))

(deftest test-convergence
  (is (converges? (range 10) 5))
  (is (not (converges? [] 5)))
  (is (not (converges? (range 10) 5 {:hits-needed 2})))
  (is (converges? (range 10) 5 {:hits-needed 2 :tolerance 1}))
  (is (not (converges? (range 10) 15)))
  (is (converges? (range 10) 15 {:tolerance 10})))