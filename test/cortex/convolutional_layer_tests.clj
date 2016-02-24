(ns cortex.convolutional-layer-tests
  (:require [clojure.core.matrix :as m]
            [cortex.protocols :as cp]
            [cortex.impl.layers :as impl]
            [cortex.core :as core]
            [cortex.backends :as b]
            [clojure.test :refer [deftest is are]]
            [clojure.pprint]))


(m/set-current-implementation :vectorz)

(def conv-layer-config (impl/create-conv-layer-config 3 3 2 2 0 0 1 1 1))

(deftest conv-rows
  (is (= (map double [1 2 4 5 2 3 5 6 4 5 7 8 5 6 8 9])
         (map double (m/eseq (impl/create-convolution-rows (range 1 10) conv-layer-config))))))


(defn create-conv-layer
  [input-dim num-channels k-dim pad stride n-kernels]
  (let [conv-config (impl/create-conv-layer-config input-dim input-dim
                                                   k-dim k-dim
                                                   pad pad
                                                   stride stride
                                                   num-channels)
        input (m/array :vectorz (flatten (map #(repeat num-channels %) (range 1 (+ (* input-dim input-dim) 1)))))
        weights (m/array :vectorz (map #(repeat (* k-dim k-dim num-channels) %) (range 1 (+ n-kernels 1))))
        bias (m/zero-array :vectorz [n-kernels])
        output-dim (impl/get-padded-strided-dimension input-dim pad k-dim stride)
        output-gradient (m/array :vectorz (repeat (* output-dim output-dim n-kernels) 1))
        weight-gradient (m/zero-array :vectorz (m/shape weights))
        bias-gradient (m/zero-array :vectorz [n-kernels])]
    (impl/map->Convolutional {:weights weights :bias bias
                              :weight-gradient weight-gradient
                              :bias-gradient bias-gradient
                              :conv-layer-config conv-config})))



(deftest basic-conv-layer
  (let [conv-layer (create-conv-layer 3 1 2 0 1 4)
        input (m/array (repeat 9 1))
        result-conv-layer (core/forward conv-layer input)
        output-gradient (m/array (flatten (repeat 4 [1 1 1 1])))
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
(def pool-layer-config (impl/create-conv-layer-config 2 2 2 2 0 0 1 1 4))

(def pool-layer-width (impl/get-padded-strided-dimension 2 0 2 1))
(def pool-layer-height (impl/get-padded-strided-dimension 2 0 2 1))
;;4 channels * output of pool layer
(def pool-layer-output-size (* 4 pool-layer-width pool-layer-height))
;;4 channels * input sizes to pool layer
 (def pool-layer-input-size (* 4 2 2))


(deftest pool-layer-basic
  (let [pool-layer (impl/map->Pooling {:output (m/zero-array :vectorz [pool-layer-output-size])
                                       :output-indexes (m/zero-array :vectorz [pool-layer-output-size])
                                       :input-gradient (m/zero-array :vectorz [pool-layer-input-size])
                                       :conv-layer-config pool-layer-config})
        input (m/array :vectorz (range 1 17))
        forward-pool-layer (core/forward pool-layer input)
        output-gradient (m/array :vectorz [1 2 3 4])
        backward-pool-layer (core/backward forward-pool-layer input output-gradient)
        input-gradient (core/input-gradient backward-pool-layer)]
    (is (= (map double (range 13 17))
           (m/eseq (core/output forward-pool-layer))))
    (is (= (map double (flatten (concat (repeat 3 [0 0 0 0])
                                        [[1 2 3 4]])))
           (m/eseq input-gradient)))
    (let [input (m/array :vectorz (range 16 0 -1))
          forward-pool-layer (core/forward backward-pool-layer input)
          output-gradient (m/array :vectorz [1 2 3 4])
          backward-pool-layer (core/backward forward-pool-layer input output-gradient)
          input-gradient (core/input-gradient backward-pool-layer)]
      (is (= (map double (range 16 12 -1))
             (m/eseq (core/output forward-pool-layer))))
      (is (= (map double (flatten (concat [[1 2 3 4]] (repeat 3 [0 0 0 0]))))
             (m/eseq input-gradient))))))




(def conv-layer-pad-config (impl/create-conv-layer-config 3 3 2 2 0 0 1 1 1))

(defn run-conv-backward
  "Run the convolutional backward pass with known input values.  Very useful
for testing against other conv net implementations."
  [input-dim num-channels k-dim pad stride n-kernels]
  (let [conv-config (impl/create-conv-layer-config input-dim input-dim
                                                   k-dim k-dim
                                                   pad pad
                                                   stride stride
                                                   num-channels)
        input (b/array (flatten (map #(repeat num-channels %)
                                     (range 1 (+ (* input-dim input-dim) 1)))))
        weights (b/array (map #(repeat (* k-dim k-dim num-channels) %)
                              (range 1 (+ n-kernels 1))))
        output-dim (impl/get-padded-strided-dimension input-dim pad k-dim stride)
        output-gradient (b/array (repeat (* output-dim output-dim n-kernels) 1))
        weight-gradient (b/new-array (m/shape weights))
        bias (b/zero-array [n-kernels])
        bias-gradient (b/new-array (m/shape bias))
        conv-layer (impl/->Convolutional weights bias
                                         weight-gradient bias-gradient
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
           (map double (flatten (repeat 4 [5 5 5 10 10 10 10 10 10 20 20 20])))))
    (is (= (m/eseq bias-gradient)
           (map double [4 4 4 4])))
    (is (= (m/eseq input-gradient)
           (map double (repeat (* 9 3) 10))))))


(defn run-conv-gradient-copy
  ([]
   (let [channels 1
         conv-config (impl/create-conv-layer-config 3 3 2 2 0 0 1 1 channels)
         input-mat (m/array :vectorz (flatten (repeat 9 (repeat channels 1))))
         input-conv (impl/create-convolution-rows input-mat conv-config)
         output-mat (m/array :vectors (flatten (repeat 9 (repeat channels 0))))]
     (impl/convolution-to-input-vector! input-conv output-mat conv-config))))


(defn run-conv-input-copy
  ([]
   (let [conv-config (impl/create-conv-layer-config 3 3 2 2 1 1 1 1 1)
         input-mat (m/array :vectorz (range 1 10))
         input-conv (m/array :vectorz (impl/create-convolution-rows (m/zero-array :vectorz [9]) conv-config))]
     (impl/input-vector-to-convolution! input-mat input-conv conv-config))))
