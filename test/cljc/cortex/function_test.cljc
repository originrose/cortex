(ns cortex.function-test
  "Tests for behaviour of mathematical function modules"
  (:require
    #?(:cljs
        [cljs.test :refer-macros [deftest is testing]]
        :clj
        [clojure.test :refer [deftest is testing]])
    [cortex.core :refer [calc output forward backward input-gradient parameters gradient parameter-count calc-output]]
    [cortex.backends :as b]
    [clojure.core.matrix :as m]
    #?(:cljs [thi.ng.ndarray.core :as nd])
    [cortex.layers :as layers]
    [cortex.util :as util]))

(m/set-current-implementation #?(:clj :vectorz :cljs :thing-ndarray))

(deftest test-logistic-module
  (testing "basic logistic functionality"
    (let [m (layers/logistic [3])]
      (is (m/equals [0 0 0] (output m)))
      (let [cm (calc m [-1000 0 1000])]
        (is (m/equals [0 0.5 1] (output cm))))))

  (testing "logistic applied to scalars"
    (let [m (layers/logistic [])]
      (let [cm (calc m 0)]
        (is (m/equals 0.5 (output cm))))))

  (testing "forward and backward pass"
    (let [m (layers/logistic [3])]

      (let [input [-1000 0 1000]
            fm (forward m input)]
        (is (m/equals [0 0.5 1] (output fm)))

        (let [bm (backward fm input [10 10 -10])]
          (is (m/equals [0 2.5 0] (input-gradient bm))))))))

(deftest test-function
  (testing "emulated logistic function"
    (let [m (layers/logistic [3])
          m2 (layers/function m/logistic
                              (fn [input output-gradient]
                                (m/mul output-gradient (util/logistic' (m/logistic input)))))]
      (let [v [-10 1 10]]
        (is (m/equals (calc-output m v) (calc-output m2 v))))
      (let [train (fn [m input target]
                    (let [m (forward m input)
                          d (m/sub (output m) target)]
                      (backward m input d)))
            input [-10 1 10]
            target [0.1 0.5 0.6]
            m (train m input target)
            m2 (train m2 input target)]
        (is (m/equals (input-gradient m) (input-gradient m2)))))))

(deftest test-linear-module
  (testing "Parameters"
    (let [wm (layers/linear [[1 2] [3 4]] [0 10])
          parm (parameters wm)
          grad (gradient wm)]
      (is (== 6 (parameter-count wm)))
      (is (m/equals [1 2 3 4 0 10] parm))
      (is (= (m/shape parm) (m/shape grad)))
      (is (m/zero-matrix? grad))))

  (testing "Calculation"
    (let [wm (layers/linear [[1 2] [3 4]] [0 10])
          wm (calc wm [1 2])]
      (is (m/equals [5 21] (output wm)))))
  (testing "Backward pass"
    (let [wm (layers/linear [[1 2] [3 4]] [0 10])
          wm (backward wm [1 2] [1 2])]
      (is (m/equals [1 2 2 4] (:weight-gradient wm)))
      (is (m/equals [1 2] (:bias-gradient wm)))
      (is (m/equals [7 10] (:input-gradient wm))))))


(deftest test-linear-fastpath-module
  (testing "Parameters"
    (let [wm (layers/linear (b/array [[1 2] [3 4]]) (b/array [0 10]))
          parm (parameters wm)
          grad (gradient wm)]
      (is (== 6 (parameter-count wm)))
      (is (m/equals [1 2 3 4 0 10] parm))
      (is (= (m/shape parm) (m/shape grad)))
      (is (m/zero-matrix? grad))))

  (testing "Calculation"
    (let [wm (layers/linear (b/array [[1 2] [3 4]]) (b/array [0 10]))
          wm (calc wm (b/array [1 2]))]
      (is (m/equals [5 21] (output wm)))))
  (testing "Backward pass"
    (let [wm (layers/linear (b/array [[1 2] [3 4]]) (b/array [0 10]))
          wm (backward wm (b/array [1 2]) (b/array [1 2]))]
      (is (m/equals [1 2 2 4] (:weight-gradient wm)))
      (is (m/equals [1 2] (:bias-gradient wm)))
      (is (m/equals [7 10] (:input-gradient wm))))))
