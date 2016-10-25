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

(defn num-random-indexes
  [num-items num-indexes]
  (vec (if num-indexes
         (take num-indexes (range num-items))
         (range num-items))))

(defn mnist-eval-dataset
  [image-count]
  (let [indexes (num-random-indexes (count @mnist/training-data) image-count)]
   (ds/->InMemoryDataset  [(mapv re-bias-mnist @mnist/training-data)
                           @mnist/training-labels]
                          {:data {:shape (ds/create-image-shape 1 28 28) :index 0}
                           :labels {:shape 10 :index 1}}
                          {:training indexes :cross-validation indexes
                           :holdout indexes :all indexes})))


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
