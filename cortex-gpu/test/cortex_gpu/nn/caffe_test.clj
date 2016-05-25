(ns cortex-gpu.nn.caffe-test
  (:require [clojure.test :refer :all]
            [cortex.nn.caffe :as caffe]
            [cortex.nn.description :as desc]
            [cortex-gpu.nn.description :as gpu-desc]
            [cortex-gpu.nn.train-test :as train-test]
            [cortex-gpu.nn.train :as train]
            [cortex-gpu.test-framework :as framework]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.macros :refer [c-for]]
            [clojure.java.io :as io]))

(use-fixtures :once framework/with-contexts)
(use-fixtures :each framework/with-resource-context)

(defn re-bias-mnist
  [^doubles input-data]
  (let [input-count (alength input-data)
        ^doubles retval (double-array input-count)]
    (c-for [idx 0 (< idx input-count) (inc idx)]
           (aset retval idx (+ (aget input-data idx) 0.5)))
    retval))


(deftest caffe-mnist
  (let [caffe-desc (caffe/caffe->description
                    (caffe/load-text-caffe-file (io/resource "lenet.prototxt"))
                    (caffe/load-binary-caffe-file (io/resource "lenet_iter_10000.caffemodel")))
        gpu-network (gpu-desc/build-and-create-network
                     caffe-desc)
        score (train/evaluate-softmax gpu-network
                                      (mapv re-bias-mnist @train-test/training-data)
                                      @train-test/training-labels)]
    (is (> score 0.98))))
