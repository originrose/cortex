(ns cortex.optimise.optimisers-test
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.core.matrix.operators :refer [+ - * /]]
            [clojure.test :refer :all]
            [cortex.optimise.optimisers :refer :all]
            [cortex.nn.protocols :as cp]
            [cortex.util :refer [def-]]))

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
           (->> ((juxt cp/parameters cp/get-state))))
         [[-9 -18 -37] {}])))
