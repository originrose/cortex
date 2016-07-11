(ns cortex.nn.convolutional-layer-tests
  (:require [clojure.core.matrix :as m]
            [cortex.nn.protocols :as cp]
            [cortex.nn.impl.layers :as impl]
            [cortex.nn.impl.layers.convolution :as conv]
            [cortex.nn.core :as core]
            [cortex.nn.backends :as b]
            [clojure.test :refer [deftest is are]]
            [clojure.pprint]))


(m/set-current-implementation :vectorz)

(def conv-layer-config (conv/create-conv-layer-config 3 3 2 2 0 0 1 1 1))

(deftest conv-rows
  (let [conv-matrix (conv/planar-input->convolution
                      (range 1 10) conv-layer-config)]
   (is (= (map double [1 2 4 5 2 3 5 6 4 5 7 8 5 6 8 9])
          (map double (m/eseq conv-matrix))))
   (let [input-mat (b/new-array [9])]
     (conv/convolution->planar-output! conv-matrix input-mat conv-layer-config)
     (is (= (map double [1 4 3 8 20 12 7 16 9])
            (m/eseq input-mat))))))


(defn create-conv-layer
  [input-dim num-channels k-dim pad stride n-kernels]
  (let [conv-config (conv/create-conv-layer-config input-dim input-dim
                                                   k-dim k-dim
                                                   pad pad
                                                   stride stride
                                                   num-channels
                                                   n-kernels)
        input (b/array  (flatten (map #(repeat num-channels %)
                                              (range 1 (+ (* input-dim input-dim) 1)))))
        weights (b/array  (map #(repeat (* k-dim k-dim num-channels) %)
                                       (range 1 (+ n-kernels 1))))
        bias (b/zero-array  [1 n-kernels])]
    (conv/->Convolutional weights bias conv-config)))



(deftest basic-conv-layer
  (let [conv-layer (create-conv-layer 3 1 2 0 1 4)
        input (b/array (repeat 9 1))
        result-conv-layer (core/forward conv-layer input)
        output-gradient (b/array (flatten (repeat 4 [1 1 1 1])))
        final-conv-layer (core/backward result-conv-layer input output-gradient)
        input-gradient (core/input-gradient final-conv-layer)]
    (is (= (repeat 16 4.0)
           (m/eseq (:weight-gradient final-conv-layer))))
    (is (= (repeat 4 4.0)
           (m/eseq (:bias-gradient final-conv-layer))))
    (is (= (map #(double (* 10 %)) [1 2 1 2 4 2 1 2 1])
           (m/eseq input-gradient)))
    (let [forward-conv-layer (core/forward final-conv-layer input)
          backward-conv-layer (core/backward forward-conv-layer input output-gradient)
          input-gradient (core/input-gradient backward-conv-layer)]
      (is (= (repeat 16 8.0)
             (m/eseq (:weight-gradient final-conv-layer))))
      (is (= (repeat 4 8.0)
             (m/eseq (:bias-gradient final-conv-layer))))
      (is (= (map #(double (* 10 %)) [1 2 1 2 4 2 1 2 1])
             (m/eseq input-gradient))))))


;;Pooling layer conceptually after the conv layer above meaning it gets a 2,2 image
;;with 4 channels.
(def pool-layer-config (conv/create-conv-layer-config 2 2 2 2 0 0 1 1 4))

(def pool-layer-width (conv/get-padded-strided-dimension :pooling 2 0 2 1))
(def pool-layer-height (conv/get-padded-strided-dimension :pooling 2 0 2 1))
;;4 channels * output of pool layer
(def pool-layer-output-size (* 4 pool-layer-width pool-layer-height))
;;4 channels * input sizes to pool layer
 (def pool-layer-input-size (* 4 2 2))


(deftest pool-layer-basic
  (let [pool-layer (conv/->Pooling pool-layer-config)
        input (b/array  (range 1 17))
        forward-pool-layer (core/forward pool-layer input)
        output-gradient (b/array  [1 2 3 4])
        backward-pool-layer (core/backward forward-pool-layer input output-gradient)
        input-gradient (core/input-gradient backward-pool-layer)]
    (is (= (map double [4 8 12 16])
           (m/eseq (core/output forward-pool-layer))))
    (is (= (map double (flatten (map #(vector 0 0 0 %) (range 1 5))))
           (m/eseq input-gradient)))
    (let [input (b/array  (range 16 0 -1))
          forward-pool-layer (core/forward backward-pool-layer input)
          output-gradient (b/array  [1 2 3 4])
          backward-pool-layer (core/backward forward-pool-layer input output-gradient)
          input-gradient (core/input-gradient backward-pool-layer)]
      (is (= (map double [16 12 8 4])
             (m/eseq (core/output forward-pool-layer))))
      (is (= (map double (flatten (map #(vector % 0 0 0) (range 1 5))))
             (m/eseq input-gradient))))))


(deftest pool-layer-negative-input
  "Test the pooling layer output when input is negative"
  (let [pool-layer (conv/->Pooling pool-layer-config)
        input (b/array (range -8 8))
        forward-pool-layer (core/forward pool-layer input)
        output-gradient (b/array  [1 2 3 4])
        backward-pool-layer (core/backward forward-pool-layer input output-gradient)
        input-gradient (core/input-gradient backward-pool-layer)]
    (is (= (map double [-5 -1 3 7])
           (m/eseq (core/output forward-pool-layer))))
    (is (= (map double (flatten (map #(vector 0 0 0 %) (range 1 5))))
           (m/eseq input-gradient)))
    (let [input (b/array  (range 8 -8 -1))
          forward-pool-layer (core/forward backward-pool-layer input)
          output-gradient (b/array  [1 2 3 4])
          backward-pool-layer (core/backward forward-pool-layer input output-gradient)
          input-gradient (core/input-gradient backward-pool-layer)]
      (is (= (map double [8 4 0 -4])
             (m/eseq (core/output forward-pool-layer))))
      (is (= (map double (flatten (map #(vector % 0 0 0) (range 1 5))))
             (m/eseq input-gradient))))))


(deftest pool-layer-negative-weights
  "Test the pooling layer output when input-size and pooling size are not exactly divisible
which causes a partial window at the boundry"
  (let [pool-layer (conv/->Pooling (conv/create-conv-layer-config 6 6 2 2 0 0 2 2 1 1))
        input (b/array (repeat 36 -11))
        forward-pool-layer (core/forward pool-layer input)]
    (is (= (map double (into [] (repeat 9 -11)))
           (m/eseq (core/output forward-pool-layer))))
    (let [pool-layer (conv/->Pooling (conv/create-conv-layer-config 6 6 3 3 0 0 2 2 1 1))
          forward-pool-layer (core/forward pool-layer input)]
      (is (= (map double (into [] (repeat 9 -11)))
           (m/eseq (core/output forward-pool-layer)))))
    (let [pool-layer (conv/->Pooling (conv/create-conv-layer-config 6 6 3 3 0 0 2 2 1 1))
          input (b/array (repeat 36 11))
          forward-pool-layer (core/forward pool-layer input)]
      (is (= (map double (into [] (repeat 9 11)))
           (m/eseq (core/output forward-pool-layer)))))))


(def conv-layer-pad-config (conv/create-conv-layer-config 3 3 2 2 0 0 1 1 1))

(defn run-conv-backward
  "Run the convolutional backward pass with known input values.  Very useful
for testing against other conv net implementations."
  [input-dim num-channels k-dim pad stride n-kernels]
  (let [conv-config (conv/create-conv-layer-config input-dim input-dim
                                                   k-dim k-dim
                                                   pad pad
                                                   stride stride
                                                   num-channels
                                                   n-kernels)
        input (b/array (flatten (map #(repeat num-channels %)
                                     (range 1 (+ (* input-dim input-dim) 1)))))
        weights (b/array (map #(repeat (* k-dim k-dim num-channels) %)
                              (range 1 (+ n-kernels 1))))
        output-dim (conv/get-padded-strided-dimension :convolutional input-dim pad k-dim stride)
        output-gradient (b/array (repeat (* output-dim output-dim n-kernels) 1))
        bias (b/zero-array [1 n-kernels])
        conv-layer (conv/->Convolutional weights bias
                                         conv-config)
        conv-layer (cp/forward conv-layer input)
        conv-layer (cp/backward conv-layer input output-gradient)
        result      {:output-gradient output-gradient
                     :input-gradient (:input-gradient conv-layer)
                     :weight-gradient (:weight-gradient conv-layer)
                     :bias-gradient (:bias-gradient conv-layer)}]
    result))


(deftest test-strided-padded-conv-backward
  (let [result (run-conv-backward 3 3 2 1 2 4)
        bias-gradient (:bias-gradient result)
        weight-gradient (:weight-gradient result)
        input-gradient (:input-gradient result)]
    (is (= (m/eseq weight-gradient)
           (map double (flatten (repeat 4 [2 4 4 8 5 10 10 20 8 16 16 32])))))
    (is (= (m/eseq bias-gradient)
           (map double [4 4 4 4])))
    (is (= (m/eseq input-gradient)
           (map double (repeat (* 9 3) 10))))))
