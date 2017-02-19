(ns cortex.optimise.parameters-test
  (:require [clojure.test :refer :all]
            [cortex.optimise.protocols :as cp]
            [cortex.optimise.parameters :refer :all]))

(def sample-map
  {:state {:params [1 2 3]
           :data [:a :b :c]}})

(deftest protocol-extension-test
  (is (= (cp/parameters sample-map)
         [1 2 3]))
  (is (= (cp/update-parameters sample-map [4 5 6])
         {:state {:params [4 5 6]
                  :data [:a :b :c]}})))
