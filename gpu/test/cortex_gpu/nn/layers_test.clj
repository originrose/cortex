(ns cortex-gpu.nn.layers-test
  (:require [clojure.test :refer :all]
            [cortex-gpu.cuda :as cuda]
            [cortex-gpu.nn.cudnn :as cudnn]
            [cortex-gpu.nn.layers :as layers]
            [cortex-gpu.nn.description :as gpu-desc]
            [cortex.nn.description :as desc]
            [cortex.nn.protocols :as cp]
            [cortex-gpu.test-framework :refer [def-double-float-test] :as framework]
            [clojure.core.matrix :as m]
            [cortex.nn.impl.layers.convolution :as conv]
            [mikera.vectorz.core]))

(use-fixtures :once framework/with-contexts)
(use-fixtures :each framework/with-resource-context)


(def-double-float-test relu-activation
  (let [item-count 10
        ;;sums to zero
        input (cudnn/array (flatten (repeat (/ item-count 2) [-1 1])))
        layer (layers/->Activation item-count cudnn/activation-relu)
        layer (cp/setup layer 1)
        layer (cp/calc layer input)
        _ (cuda/check-errors)
        output (cp/output layer)
        output-data (cudnn/to-double-array output)]
    (is (= (double (/ item-count 2))
           (m/esum output-data)))

    (let [output-gradient (cudnn/array (repeat item-count 1))
          layer (cp/backward layer input output-gradient)
          _ (cuda/check-errors)
          input-gradient (cp/input-gradient layer)
          input-grad-data (cudnn/to-double-array input-gradient)]
      (is (= (double (/ item-count 2))
             (m/esum input-grad-data))))))

(def-double-float-test relu-activation-batch
  (let [item-count 10000
        items-per-batch 5
        ;;sums to zero
        input (cudnn/array (flatten (repeat (* items-per-batch
                                               (/ item-count 2)) [-1 1]))
                           items-per-batch)
        layer (layers/->Activation item-count cudnn/activation-relu)
        layer (cp/setup layer items-per-batch)
        layer (cp/calc layer input)
        _ (cuda/check-errors)
        output (cp/output layer)
        output-data (cudnn/to-double-array output)]

    (is (= (double (* items-per-batch
                      (/ item-count 2)))
           (m/esum output-data)))

    (let [output-gradient (cudnn/array (repeat (* items-per-batch item-count) 1)
                                       items-per-batch)
          layer (cp/backward layer input output-gradient)
          _ (cuda/check-errors)
          input-gradient (cp/input-gradient layer)
          input-grad-data (cudnn/to-double-array input-gradient)]
      (is (= (double (* items-per-batch
                        (/ item-count 2)))
             (m/esum input-grad-data))))))



(def-double-float-test linear
  (let [weights (cudnn/array [[1 2] [3 4]])
        bias (cudnn/array [0 10])
        input (cudnn/array [1 2])
        layer (layers/->Linear weights bias nil)
        layer (cp/setup layer 1)
        layer (cp/calc layer input)
        output (cp/output layer)
        output-data (cudnn/to-double-array output)]
    (is (= (map double [5 21])
           (m/eseq output-data)))
    (let [output-gradient (cudnn/array [1 2])
          layer (cp/backward layer input output-gradient)
          weight-gradient (vec (cudnn/to-double-array (:weight-gradient layer)))
          bias-gradient (vec (cudnn/to-double-array (:bias-gradient layer)))
          input-gradient (vec (cudnn/to-double-array (:input-gradient layer)))]
      (is (m/equals [1 2 2 4] weight-gradient))
      (is (m/equals [1 2] bias-gradient))
      (is (m/equals [7 10] input-gradient)))))


(def-double-float-test linear-batch
  (let [num-batch-items 10
        weights (cudnn/array [[1 2] [3 4]])
        bias (cudnn/array [0 10])
        input (cudnn/array (flatten (repeat num-batch-items [1 2])) num-batch-items)
        layer (layers/->Linear weights bias nil)
        layer (cp/setup layer num-batch-items)
        layer (cp/calc layer input)
        output (cp/output layer)
        output-data (cudnn/to-double-array output)]
    (is (= (map double (flatten (repeat num-batch-items [5 21])))
           (m/eseq output-data)))
    (let [output-gradient (cudnn/array (flatten (repeat num-batch-items [1 2]))
                                       num-batch-items)
          layer (cp/backward layer input output-gradient)
          weight-gradient (vec (cudnn/to-double-array (:weight-gradient layer)))
          bias-gradient (vec (cudnn/to-double-array (:bias-gradient layer)))
          input-gradient (vec (cudnn/to-double-array (:input-gradient layer)))]
      (is (m/equals (mapv #(* % num-batch-items) [1 2 2 4]) weight-gradient))
      (is (m/equals (mapv #(* % num-batch-items) [1 2]) bias-gradient))
      (is (m/equals (flatten (repeat num-batch-items [7 10])) input-gradient)))))


(def-double-float-test l2-max-constraint
  (let [input-size 100
        output-size 100
        weight-matrix (cudnn/array (partition output-size (range (* input-size output-size))))
        weight-clone-temp (cudnn/new-array (cudnn/shape weight-matrix))
        weight-magnitude-temp (cudnn/new-array [output-size])
        ones-vec (cudnn/allocate-ones output-size)]
    (cudnn/apply-l2-max-constraint weight-matrix weight-clone-temp weight-magnitude-temp
                                   ones-vec 1.0)
    (let [weights (cudnn/to-double-array weight-matrix)
          double-mat (m/reshape weights [output-size input-size])
          magnitudes (map m/magnitude (m/rows double-mat))
          mag-sum (m/esum magnitudes)]
      (is (framework/about-there? mag-sum output-size 0.0001)))))


(deftest softmax
  (let [input (cudnn/array (vec (take 10 (flatten (repeat [1 2 3 4])))))
        layer (layers/softmax 10)
        layer (cp/setup layer 1)
        layer (cp/calc layer input)
        output (cp/output layer)
        output-data (cudnn/to-double-array output)]
    (is (framework/about-there? [0.015127670383492609,0.041121271510366035,0.11177920510975863
                                 ,0.30384738204945333,0.015127670383492609,0.041121271510366035
                                 ,0.11177920510975863,0.30384738204945333,0.015127670383492609
                                 ,0.041121271510366035] output-data))
    (let [output-gradient (cudnn/array (repeat 10 1))
          layer (cp/backward layer input output-gradient)
          input-gradient (cudnn/to-double-array (cp/input-gradient layer))]
      (is (= (map double (repeat 10 1))
             (seq input-gradient))))))


(def-double-float-test softmax-batch
  (let [batch-count 10
        input (cudnn/array (vec (flatten (repeat batch-count
                                                 (take 10 (flatten (repeat [1 2 3 4]))))))
                           batch-count)
        layer (layers/softmax 10)
        layer (cp/setup layer batch-count)
        layer (cp/calc layer input)
        output (cp/output layer)
        output-data (cudnn/to-double-array output)]
    (is (framework/about-there?
         (flatten (repeat batch-count
                          [0.015127670383492609,0.041121271510366035,0.11177920510975863
                           ,0.30384738204945333,0.015127670383492609,0.041121271510366035
                           ,0.11177920510975863,0.30384738204945333,0.015127670383492609
                           ,0.041121271510366035])) output-data))
    (let [output-gradient (cudnn/array (repeat (* batch-count 10) 1))
          layer (cp/backward layer input output-gradient)
          input-gradient (cudnn/to-double-array (cp/input-gradient layer))]
      (is (= (map double (repeat (* 10 batch-count) 1))
             (seq input-gradient))))))


(def-double-float-test multi-channel-softmax-batch
  (let [batch-count 2
        channel-count 4
        item-count 10
        input (cudnn/array (vec (flatten (repeat batch-count (repeat channel-count
                                                                       (take item-count (flatten (repeat [1 2 3 4])))))))
                           batch-count)
        layer (layers/softmax item-count channel-count)
        layer (cp/setup layer batch-count)
        layer (cp/calc layer input)
        output (cp/output layer)
        output-data (cudnn/to-double-array output)]
    (is (framework/about-there?
         (flatten (repeat batch-count
                          (repeat channel-count (take item-count [0.015127670383492609,0.041121271510366035,0.11177920510975863
                                                                  ,0.30384738204945333,0.015127670383492609,0.041121271510366035
                                                                  ,0.11177920510975863,0.30384738204945333,0.015127670383492609
                                                                  ,0.041121271510366035])))) output-data))
    (let [output-gradient (cudnn/array (repeat (* batch-count channel-count item-count) 1))
          layer (cp/backward layer input output-gradient)
          input-gradient (cudnn/to-double-array (cp/input-gradient layer))]
      (is (= (map double (repeat (* item-count batch-count channel-count) 1))
             (seq input-gradient))))))


(def-double-float-test adadelta-test
  (let [decay 0.05
        epsilon 1e-6
        grad-sq-accum (cudnn/array [1 2 3])
        dx-sq-accum (cudnn/array [1 2 3])
        gradients (cudnn/array [1 1 1])
        parameters (cudnn/array [1 1 1])
        grad-accum-answer [1.0,1.95,2.8999999999999995]
        dx-accum-answer [1.0,1.9512820512754767,2.901724137925089]
        params-answer [0.0,-0.012739367018747672,-0.01709525537276191]]
    (cudnn/adadelta-step decay epsilon
                         grad-sq-accum dx-sq-accum
                         1.0 gradients parameters)
    (is (framework/about-there? (cudnn/to-double-array parameters) params-answer))
    (is (framework/about-there? (cudnn/to-double-array grad-sq-accum) grad-accum-answer))
    (is (framework/about-there? (cudnn/to-double-array dx-sq-accum) dx-accum-answer))))


(defn create-conv-layer
  [input-dim num-channels k-dim pad stride n-kernels]
  (let [conv-config (conv/create-conv-layer-config input-dim input-dim
                                                   k-dim k-dim
                                                   pad pad
                                                   stride stride
                                                   num-channels
                                                   n-kernels)
        weights (cudnn/array (map #(repeat (* k-dim k-dim num-channels) %)
                                  (range 1 (+ n-kernels 1))))
        bias (cudnn/zero-array [n-kernels])]
    (layers/->Convolutional weights bias conv-config nil)))


(def-double-float-test basic-conv-layer
  (let [batch-size 2
        channel-count 4
        conv-layer (create-conv-layer 3 1 2 0 1 channel-count)
        input (cudnn/array (repeat (* 9 batch-size) 1) batch-size)
        output-gradient (cudnn/array (flatten
                                      (repeat (* 4 batch-size) [1 1 1 1])) batch-size)
        conv-layer (cp/setup conv-layer batch-size)
        conv-layer (cp/forward conv-layer input)
        conv-layer (cp/backward conv-layer input output-gradient)
        input-gradient (cp/input-gradient conv-layer)]
    (is (= (flatten (repeat batch-size (mapcat #(vector % % % %)
                                               (map double [4 8 12 16]))))
           (seq (cudnn/to-double-array (cp/output conv-layer)))))
    (is (= (repeat 16 8.0)
           (m/eseq (cudnn/to-double-array (:weight-gradient conv-layer)))))
    (is (= (repeat 4 8.0)
           (m/eseq (cudnn/to-double-array (:bias-gradient conv-layer)))))
    (is (= (flatten (repeat batch-size (map #(double (* 10 %)) [1 2 1 2 4 2 1 2 1])))
           (m/eseq (cudnn/to-double-array input-gradient))))))


(def pool-layer-config (conv/create-conv-layer-config 2 2 2 2 0 0 1 1 4))


(def-double-float-test pool-layer-basic
  (let [batch-size 2
        pool-layer (layers/->Pooling pool-layer-config)
        pool-layer (cp/setup pool-layer batch-size)
        input (cudnn/array (flatten (repeat batch-size (range 1 17))) batch-size)
        output-gradient (cudnn/array (flatten (repeat batch-size [1 2 3 4])) batch-size)
        pool-layer (cp/forward pool-layer input)
        pool-layer (cp/backward pool-layer input output-gradient)
        output (cp/output pool-layer)
        input-gradient (cp/input-gradient pool-layer)]
    (is (= (map double [4 8 12 16 4 8 12 16])
           (m/eseq (cudnn/to-double-array output))))
    (is (= (map double (flatten (repeat batch-size (map #(vector 0 0 0 %) (range 1 5)))))
           (m/eseq (cudnn/to-double-array input-gradient))))
    (let [input (cudnn/array  (repeat batch-size (range 16 0 -1)) batch-size)
          output-gradient (cudnn/array (flatten (repeat batch-size  [1 2 3 4])) batch-size)
          pool-layer (cp/forward pool-layer input)
          pool-layer (cp/backward pool-layer input output-gradient)
          input-gradient (cp/input-gradient pool-layer)]
      (is (= (map double (flatten (repeat batch-size [16 12 8 4])))
             (m/eseq (cudnn/to-double-array (cp/output pool-layer)))))
      (is (= (map double (flatten (repeat batch-size (map #(vector % 0 0 0) (range 1 5)))))
             (m/eseq (cudnn/to-double-array input-gradient)))))))

(defn count-zeros
  [item-seq]
  (count (filter #(= 0.0 (double %)) item-seq)))


(def-double-float-test dropout-constant
  (let [batch-size 5
        item-count 20
        input (cudnn/array (repeat (* batch-size item-count) 1.0) batch-size)
        output-gradient (cudnn/array (repeat (* batch-size item-count) 2.0) batch-size)
        dropout-layer (layers/dropout item-count 0.8)
        dropout-layer (cp/setup dropout-layer batch-size)
        repeat-count 30
        answer-seq
        (doall
         (for [iter (range repeat-count)]
           (let [dropout-layer (cp/prepare-forward dropout-layer)
                 dropout-layer (cp/forward dropout-layer input)
                 dropout-layer (cp/backward dropout-layer input output-gradient)
                 output (seq (cudnn/to-double-array (cp/output dropout-layer)))
                 input-gradient (seq (cudnn/to-double-array (cp/input-gradient dropout-layer)))]
             [(m/esum output) (count-zeros output)
              (m/esum input-gradient) (count-zeros input-gradient)])))
        final-aggregate  (reduce m/add answer-seq)
        final-answer (m/div final-aggregate repeat-count)
        total-elem-count (double (* item-count batch-size))]
    ;;zero count should be identical
    (is (= (final-answer 1) (final-answer 3)))
    (is (framework/about-there? (final-answer 0) total-elem-count 2))
    (is (framework/about-there? (final-answer 2) (* 2.0 total-elem-count) 4))))



(def-double-float-test dropout-multiplicative
  (let [batch-size 5
        item-count 100
        input (cudnn/array (repeat (* batch-size item-count) 1.0) batch-size)
        output-gradient (cudnn/array (repeat (* batch-size item-count) 2.0) batch-size)
        dropout-layer (layers/dropout item-count 1.0 cudnn/dropout-type-multiplicative)
        dropout-layer (cp/setup dropout-layer batch-size)
        dropout-layer (cp/prepare-forward dropout-layer)
        repeat-count 30
        answer-seq
        (doall
         (for [iter (range repeat-count)]
           (let [dropout-layer (cp/prepare-forward dropout-layer)
                 dropout-layer (cp/forward dropout-layer input)
                 dropout-layer (cp/backward dropout-layer input output-gradient)
                 output (seq (cudnn/to-double-array (cp/output dropout-layer)))
                 input-gradient (seq (cudnn/to-double-array (cp/input-gradient dropout-layer)))]
             [(m/esum output) (m/esum input-gradient)])))
        final-aggregate  (reduce m/add answer-seq)
        final-answer (m/div final-aggregate repeat-count)
        total-elem-count (double (* item-count batch-size))]
    (is (framework/about-there? (final-answer 0) total-elem-count 10))
    (is (framework/about-there? (final-answer 1) (* 2.0 total-elem-count) 20))))


(deftest split-basic
  (let [item-count 1000
        items-per-batch 5
        ;;sums to zero
        input (cudnn/array (flatten (repeat (* items-per-batch
                                               (/ item-count 2)) [-1 1]))
                           items-per-batch)
        desc [(desc/input item-count)
              (desc/split [[(desc/relu)] [(desc/relu)]])]
        layer (gpu-desc/build-and-create-network desc)
        layer (cp/setup layer items-per-batch)
        layer (cp/multi-forward layer [input])
        output (cp/multi-output layer)
        _ (is (= 2 (count output)))
        output-data (mapv cudnn/to-double-array output)]

    (is (every? #(= (double (* items-per-batch
                               (/ item-count 2)))
                    %)
                (map m/esum output-data)))

    (let [output-gradient (cudnn/array (repeat (* items-per-batch item-count) 1)
                                       items-per-batch)
          output-gradient [output-gradient output-gradient]
          layer (cp/multi-backward layer [input] output-gradient)
          _ (cuda/check-errors)
          input-gradient (first (cp/multi-input-gradient layer))
          input-grad-data (cudnn/to-double-array input-gradient)]
      (is (= (double (* items-per-batch item-count))
             (m/esum input-grad-data))))))
