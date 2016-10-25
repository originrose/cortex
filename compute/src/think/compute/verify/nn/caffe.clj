(ns think.compute.verify.nn.caffe
  (:require [cortex.nn.caffe :as caffe]
            [think.compute.nn.description :as desc]
            [clojure.java.io :as io]
            [think.compute.nn.train :as train]
            [clojure.test :refer :all]
            [think.compute.verify.nn.mnist :as mnist]
            [cortex.dataset :as ds]
            [clojure.core.matrix.macros :refer [c-for]]
            [think.compute.optimise :as opt]
            [cortex.nn.protocols :as cp]
            [think.compute.nn.evaluate :as nn-eval]))

(defn re-bias-mnist
  [^doubles input-data]
  (let [input-count (alength input-data)
        ^doubles retval (double-array input-count)]
    (c-for [idx 0 (< idx input-count) (inc idx)]
           (aset retval idx (+ (aget input-data idx) 0.5)))
    retval))


(defn mnist-eval-dataset
  [image-count]
  (ds/take-n image-count (mnist/mnist-dataset :data-transform-function re-bias-mnist)))


(defn caffe-mnist
  [backend & {:keys [image-count]}]
  (let [caffe-desc (caffe/caffe->description
                    (caffe/load-text-caffe-file (io/resource "lenet.prototxt"))
                    (caffe/load-binary-caffe-file (io/resource "lenet_iter_10000.caffemodel")))
        batch-size 10
        net (desc/build-and-create-network caffe-desc backend batch-size)
        dataset (mnist-eval-dataset image-count)
        score (nn-eval/evaluate-softmax net dataset [:data] :batch-type :holdout :dataset-label-name :labels)]
    (is (> score 0.98))))
