(ns cortex.tensor-dimensions-test
  (:require [cortex.tensor :as ct]
            [clojure.test :refer :all]))



(deftest dimensions-in-place-reshape-test
  (is (= {:shape [6 2]
          :strides [2 1]}
         (ct/dimensions-in-place-reshape
          (ct/dimensions [3 2 2] :strides [4 2 1]) [6 2])))
  (is (= {:shape [3 4]
          :strides [4 1]}
         (ct/dimensions-in-place-reshape
          (ct/dimensions [3 2 2] :strides [4 2 1])
          [3 4])))
  (is (= {:shape [3 4]
          :strides [5 1]}
         (ct/dimensions-in-place-reshape
          (ct/dimensions [3 2 2] :strides [5 2 1])
          [3 4])))
  (is (= {:shape [20 8] :strides [10 1]}
         (ct/dimensions-in-place-reshape
          {:shape [4 5 8] :strides [50 10 1]}
          [20 8]))))
