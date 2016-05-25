(ns cortex-datasets.math-test
  (:require [clojure.test :refer :all]
            [cortex-datasets.math :as math]
            [clojure.core.matrix :as m]
            [mikera.vectorz.matrix-api]))

(m/set-current-implementation :vectorz)  ;; use Vectorz as default matrix implementation


(deftest normalize-test
  (let [num-rows 10
        num-cols 20
        test-mat (m/array (partition num-cols (range (* num-rows num-cols))))]
    (math/normalize-data! test-mat)
    (let [means (math/matrix-col-means test-mat)
          stddevs (math/matrix-col-stddevs test-mat)]

      (is (math/very-near? (m/esum means) 0))
      (is (math/very-near? (m/esum stddevs) num-cols)))))


(deftest parallel-normalize-test
  (let [num-rows 200
        num-cols 100
        test-mat (m/array (partition num-cols (range (* num-rows num-cols))))
        answer-mat (m/clone test-mat)
        parallel-answers (math/parallel-normalize-data! test-mat)
        answers (math/normalize-data! answer-mat)
        means (math/matrix-col-means (:data parallel-answers))
        stddevs (math/matrix-col-stddevs (:data parallel-answers))]
    (is (m/equals (:means parallel-answers) (:means answers)))
    (is (m/equals (:stddevs parallel-answers) (:stddevs answers)))
    (is (math/very-near? (m/esum means) 0.0))
    (is (math/very-near? (m/esum stddevs) num-cols))))
