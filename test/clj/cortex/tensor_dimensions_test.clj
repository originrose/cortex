(ns cortex.tensor-dimensions-test
  (:require [cortex.tensor.dimensions :as ct-dims]
            [clojure.test :refer :all]))



(deftest dimensions-in-place-reshape-test
  (is (= {:shape [6 2]
          :strides [2 1]}
         (ct-dims/in-place-reshape
          (ct-dims/dimensions [3 2 2] :strides [4 2 1]) [6 2])))
  (is (= {:shape [3 4]
            :strides [4 1]}
           (ct-dims/in-place-reshape
            (ct-dims/dimensions [3 2 2] :strides [4 2 1])
            [3 4])))
  (is (= {:shape [3 4]
          :strides [5 1]}
         (ct-dims/in-place-reshape
          (ct-dims/dimensions [3 2 2] :strides [5 2 1])
          [3 4])))
  (is (= {:shape [20 8] :strides [10 1]}
         (ct-dims/in-place-reshape
          {:shape [4 5 8] :strides [50 10 1]}
          [20 8])))
  (is (= {:shape [20 8 1 1] :strides [10 1 1 1]}
         (ct-dims/in-place-reshape
          {:shape [4 5 8] :strides [50 10 1]}
          [20 8 1 1])))
  (is (= {:shape [1 1 20 8] :strides [200 200 10 1]}
         (ct-dims/in-place-reshape
          {:shape [4 5 8] :strides [50 10 1]}
          [1 1 20 8])))
  (is (= {:shape [169 5] :strides [5 1]}
         (ct-dims/in-place-reshape
          {:shape [845] :strides [1]}
          [169 5])))
  ;;This test is just f-ed up.  But the thing is that if the dimensions are dense, then in-place reshape
  ;;that preserves ecount is possible; it is just an arbitrary reinterpretation of the data.
  (is (= {:shape [10 4 9] :strides [36 9 1]}
         (ct-dims/in-place-reshape
          {:shape [10 1 18 2] :strides [36 36 2 1]}
          [10 4 9])))
  (is (= {:shape [845 1] :strides [25 1]}
         (ct-dims/in-place-reshape
          {:shape [13 13 5 1], :strides [1625 125 25 1]}
          [845 1])))
  (is (= {:shape [1 1] :strides [1 1]}
         (ct-dims/in-place-reshape
          {:shape [1], :strides [1]}
          [1 1]))))
