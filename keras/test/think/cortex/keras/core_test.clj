(ns think.cortex.keras.core-test
  (:require [clojure.test :refer :all]
            [think.cortex.keras.core :as keras]
            [think.compute.datatype :as dtype]
            [clojure.java.io :as io]
            [clojure.core.matrix.macros :refer [c-for]]
            [mikera.image.core :as imagez])
  (:import [java.nio ByteBuffer]))


(deftest load-vgg-16-model
  (let [desc (-> (keras/read-model "models/vgg16/model.json")
                 (keras/model->simple-description))
        type-counts (frequencies (map :type desc))
        ]
    (is (= type-counts {:convolutional 13 :input 1 :max-pooling 5 :relu 13}))
    (is (= (first desc) {:output-channels 3 :output-height 224 :output-width 224
                         :output-size 150528 :type :input}))
    (is (= (nth desc 3)
           {:bias nil :id :conv1_2 :kernel-height 3 :kernel-width 3 :l2-max-constraint nil
            :num-kernels 64 :pad-x 1 :pad-y 1 :stride-x 1 :stride-y 1 :type :convolutional
            :weights nil}))))

(deftest verify-vgg-16
  (let []))
