(ns cortex.datasets.math-test
  (:require [clojure.test :refer :all]
            [clojure.core.matrix :as m]
            [mikera.vectorz.matrix-api]
            [cortex.util :as util]))

(m/set-current-implementation :vectorz)


(deftest normalize-test
  (let [num-rows 10
        num-cols 20
        test-mat (m/array (partition num-cols (range (* num-rows num-cols))))]
    (util/normalize-data! test-mat)
    (let [means (util/matrix-col-means test-mat)
          stddevs (util/matrix-col-stddevs test-mat)]

      (is (util/very-near? (m/esum means) 0))
      (is (util/very-near? (m/esum stddevs) num-cols)))))


(deftest parallel-normalize-test
  (let [num-rows 200
        num-cols 100
        test-mat (m/array (partition num-cols (range (* num-rows num-cols))))
        answer-mat (m/clone test-mat)
        parallel-answers (util/parallel-normalize-data! test-mat)
        answers (util/normalize-data! answer-mat)
        means (util/matrix-col-means (:data parallel-answers))
        stddevs (util/matrix-col-stddevs (:data parallel-answers))]
    (is (m/equals (:means parallel-answers) (:means answers)))
    (is (m/equals (:stddevs parallel-answers) (:stddevs answers)))
    (is (util/very-near? (m/esum means) 0.0))
    (is (util/very-near? (m/esum stddevs) num-cols))))
