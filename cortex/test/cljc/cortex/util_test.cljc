(ns cortex.util-test
  (:require
    #?(:cljs [cljs.test :refer-macros [deftest is testing]]
       :clj [clojure.test :refer [deftest is testing]])
    [clojure.core.matrix :as m]
    [cortex.util :refer :all]))

(deftest confusion-test
  (let [cf (confusion-matrix ["cat" "dog" "rabbit"])
        cf (-> cf
             (add-prediction "dog" "cat")
             (add-prediction "dog" "cat")
             (add-prediction "cat" "cat")
             (add-prediction "cat" "cat")
             (add-prediction "rabbit" "cat")
             (add-prediction "dog" "dog")
             (add-prediction "cat" "dog")
             (add-prediction "rabbit" "rabbit")
             (add-prediction "cat" "rabbit")
             )]
    ;; (print-confusion-matrix cf)
    (is (= 2 (get-in cf ["cat" "dog"])))))

(deftest test-convergence
  (is (converges? (range 10) 5))
  (is (not (converges? [] 5)))
  (is (not (converges? (range 10) 5 {:hits-needed 2})))
  (is (converges? (range 10) 5 {:hits-needed 2 :tolerance 1}))
  (is (not (converges? (range 10) 15)))
  (is (converges? (range 10) 15 {:tolerance 10})))

(deftest test-mse-gardient
  (is (m/equals [-2 0 2] (mse-gradient-fn [10 11 12] [11 11 11]))))
