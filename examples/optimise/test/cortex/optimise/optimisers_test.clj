(ns cortex.optimise.optimisers-test
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.core.matrix.operators :refer [+ - * /]]
            [clojure.test :refer :all]
            [cortex.optimise.optimisers :refer :all]
            [cortex.optimise.protocols :as cp]
            [cortex.util :refer [approx= def-]]))

;;;; Protocol extensions

(def- test-optimiser-map
  {:initialize (fn [param-count]
                 {:velocity (vec (repeat param-count 0))})
   :update (fn [{:keys [params velocity]} gradient]
             {:params (+ params velocity)
              :velocity (+ velocity gradient)})})

(def- test-optimiser-fn
  (fn [params gradient]
    (+ params gradient)))

(deftest protocol-extension-test
  (testing "map as optimiser"
    (is (= (->> test-optimiser-map
             (iterate #(cp/compute-parameters % [1 2 3] (or (cp/parameters %)
                                                            [0 0 0])))
             (map (juxt cp/parameters cp/get-state))
             (rest)
             (take 3))
           [[[0 0 0] {:velocity [1 2 3]}]
            [[1 2 3] {:velocity [2 4 6]}]
            [[3 6 9] {:velocity [3 6 9]}]])))
  (testing "fn as optimiser"
    (is (= (->> test-optimiser-fn
             (iterate #(cp/compute-parameters % [1 2 3] (or (cp/parameters %)
                                                            [0 0 0])))
             (map (juxt cp/parameters cp/get-state))
             (rest)
             (take 3))
           [[[1 2 3] {}]
            [[2 4 6] {}]
            [[3 6 9] {}]]))))

;;;; Clojure implementations

;;; Ideally, there would be a standard way to mutate the internal state of an
;;; optimiser, so that this code didn't have to rely on optimisers being of
;;; particular types. But since they are maps, it's easy to do the mutation
;;; directly, using assoc.

(deftest sgd-clojure-test
  (is (= (-> (sgd-clojure :learning-rate 5)
           (cp/compute-parameters [2 4 8] [1 2 3])
           ((juxt cp/parameters cp/get-state)))
         [[-9 -18 -37] {}])))

(deftest adadelta-clojure-test
  (is (approx= 1e-8
               (-> (adadelta-clojure :decay-rate 0.5
                                     :conditioning 1)
                 (assoc-in [:state :acc-gradient] [6 3 9])
                 (assoc-in [:state :acc-step] [7 8 7])
                 (dissoc :initialize)
                 (cp/compute-parameters [2 4 8] [1 2 3])
                 ((juxt cp/parameters cp/get-state)))
               ;; The following was computed manually using a calculator.
               [[-1.309401077 -1.703280399 -0.6950417228]
                {:acc-gradient [5.0 9.5 36.5]
                 :acc-step [6.166666667 10.85714286 10.32666667]}])))

(deftest adam-clojure-test
  (is (approx= 1e-8
               (-> (adam-clojure :step-size 5
                                 :first-moment-decay 0.75
                                 :second-moment-decay 0.3
                                 :conditioning 0.2)
                 (assoc-in [:state :first-moment] [5 2 -3])
                 (assoc-in [:state :second-moment] [-1 -3 -8])
                 (assoc-in [:state :num-steps] 2)
                 (dissoc :initialize)
                 (cp/compute-parameters [5 3 7] [1 2 -5])
                 ((juxt cp/parameters cp/get-state)))
               ;; The following was computed manually using a calculator.
               [[-8.81811015 -5.613809809 -4.270259223]
                {:first-moment [5.0 2.25 -0.5]
                 :second-moment [17.2 5.4 31.9]
                 :num-steps 3}])))
