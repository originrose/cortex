(ns think.compute.verify.nn.layers
  (:require [clojure.test :refer :all]
            [think.compute.nn.backend :as nn-backend]
            [think.compute.nn.layers :as layers]
            [think.compute.math :as math]
            [cortex.nn.protocols :as cp]
            [clojure.core.matrix :as m]
            [think.compute.verify.utils :as utils]
            [cortex.nn.impl.layers.convolution :as conv]
            [cortex.util :as cu]
            [think.resource.core :as resource]))


(defn test-relu-activation
  [backend]
  (let [item-count 10
        ;;sums to zero
        input (nn-backend/array backend (flatten (repeat (/ item-count 2) [-1 1])))
        layer (layers/activation backend item-count :relu)
        layer (cp/setup layer 1)
        layer (cp/calc layer input)
        output (cp/output layer)
        output-data (nn-backend/to-double-array backend output)]
    (is (= (double (/ item-count 2))
           (m/esum output-data)))
    (let [output-gradient (nn-backend/array backend (repeat item-count 1))
          layer (cp/backward layer input output-gradient)
          input-gradient (cp/input-gradient layer)
          input-grad-data (nn-backend/to-double-array backend input-gradient)]
      (is (= (double (/ item-count 2))
             (m/esum input-grad-data))))))


(defn test-relu-activation-batch
  [backend]
  (let [item-count 10000
        items-per-batch 5
        ;;sums to zero
        input (nn-backend/array backend (flatten (repeat (* items-per-batch
                                                     (/ item-count 2)) [-1 1]))
                             items-per-batch)
        layer (layers/activation backend item-count :relu)
        layer (cp/setup layer items-per-batch)
        layer (cp/calc layer input)
        output (cp/output layer)
        output-data (nn-backend/to-double-array backend output)]

    (is (= (double (* items-per-batch
                      (/ item-count 2)))
           (m/esum output-data)))

    (let [output-gradient (nn-backend/array backend (repeat (* items-per-batch item-count) 1)
                                         items-per-batch)
          layer (cp/backward layer input output-gradient)
          input-gradient (cp/input-gradient layer)
          input-grad-data (nn-backend/to-double-array backend input-gradient)]
      (is (= (double (* items-per-batch
                        (/ item-count 2)))
             (m/esum input-grad-data))))))


(defn test-linear
  [backend]
  (let [weights (nn-backend/array backend [[1 2] [3 4]])
        bias (nn-backend/array backend [0 10])
        input (nn-backend/array backend [1 2])
        layer (layers/->Linear backend weights bias nil)
        layer (cp/setup layer 1)
        layer (cp/calc layer input)
        output (cp/output layer)
        output-data (nn-backend/to-double-array backend output)]
    (is (= (map double [5 21])
           (m/eseq output-data)))
    (let [output-gradient (nn-backend/array backend [1 2])
          layer (cp/backward layer input output-gradient)
          weight-gradient (vec (nn-backend/to-double-array backend (:weight-gradient layer)))
          bias-gradient (vec (nn-backend/to-double-array backend (:bias-gradient layer)))
          input-gradient (vec (nn-backend/to-double-array backend (:input-gradient layer)))]
      (is (m/equals [1 2 2 4] weight-gradient))
      (is (m/equals [1 2] bias-gradient))
      (is (m/equals [7 10] input-gradient)))))



(defn test-l2-max-constraint
  [backend]
  (let [input-size 100
        output-size 10
        l2-max-constraint 1.0
        weight-matrix (nn-backend/array backend
                                     (partition input-size (range (* input-size output-size))))
        bias (nn-backend/new-array backend [output-size])
        layer (layers/->Linear backend weight-matrix bias l2-max-constraint)
        layer (cp/setup layer 10)]
    (layers/apply-l2-max-constraint layer)
    (let [weights (nn-backend/to-double-array backend weight-matrix)
          double-mat (m/reshape weights [output-size input-size])
          magnitudes (map m/magnitude (m/rows double-mat))
          mag-sum (m/esum magnitudes)]
      (is (utils/about-there? mag-sum output-size 0.0001)))))


(defn test-linear-batch
  [backend]
  (let [num-batch-items 10
        weights (nn-backend/array backend [[1 2] [3 4]])
        bias (nn-backend/array backend [0 10])
        input (nn-backend/array backend (flatten (repeat num-batch-items [1 2])) num-batch-items)
        layer (layers/->Linear backend weights bias nil)
        layer (cp/setup layer num-batch-items)
        layer (cp/calc layer input)
        output (cp/output layer)
        output-data (nn-backend/to-double-array backend output)]
    (is (= (map double (flatten (repeat num-batch-items [5 21])))
           (m/eseq output-data)))
    (let [output-gradient (nn-backend/array backend (flatten (repeat num-batch-items [1 2]))
                                         num-batch-items)
          layer (cp/backward layer input output-gradient)
          weight-gradient (vec (nn-backend/to-double-array backend (:weight-gradient layer)))
          bias-gradient (vec (nn-backend/to-double-array backend (:bias-gradient layer)))
          input-gradient (vec (nn-backend/to-double-array backend (:input-gradient layer)))]
      (is (m/equals (mapv #(* % num-batch-items) [1 2 2 4]) weight-gradient))
      (is (m/equals (mapv #(* % num-batch-items) [1 2]) bias-gradient))
      (is (m/equals (flatten (repeat num-batch-items [7 10])) input-gradient)))))


(def activation-answers
  {:sigmoid [[0.2689414213699951 0.7310585786300049 0.2689414213699951 0.7310585786300049
              0.2689414213699951 0.7310585786300049 0.2689414213699951 0.7310585786300049
              0.2689414213699951 0.7310585786300049]
             [-0.19661193324148185 0.19661193324148185 -0.19661193324148185 0.19661193324148185
              -0.19661193324148185 0.19661193324148185 -0.19661193324148185 0.19661193324148185
              -0.19661193324148185 0.19661193324148185]]
   :tanh [[-0.7615941559557649 0.7615941559557649 -0.7615941559557649 0.7615941559557649
           -0.7615941559557649 0.7615941559557649 -0.7615941559557649 0.7615941559557649
           -0.7615941559557649 0.7615941559557649]
          [-0.41997434161402614 0.41997434161402614 -0.41997434161402614 0.41997434161402614
           -0.41997434161402614 0.41997434161402614 -0.41997434161402614 0.41997434161402614
           -0.41997434161402614 0.41997434161402614]]})


(defn test-activation
  [backend act-type]
  (let [item-count 10
        ;;sums to zero
        input (nn-backend/array backend (flatten (repeat (/ item-count 2) [-1 1])))
        layer (layers/activation backend item-count act-type)
        layer (cp/setup layer 1)
        layer (cp/calc layer input)
        output (cp/output layer)
        output-data (nn-backend/to-double-array backend output)]
    (is (utils/about-there? output-data (first (activation-answers act-type))))
    (let [output-gradient (nn-backend/array backend (flatten (repeat (/ item-count 2) [-1 1])))
          layer (cp/backward layer input output-gradient)
          input-gradient (cp/input-gradient layer)
          input-grad-data (nn-backend/to-double-array backend input-gradient)]
      (is (utils/about-there? input-grad-data (second (activation-answers act-type)))))))


(def activation-batch-answers nil)

(def activation-batch-size 5)

(def activation-batch-answers
  {:sigmoid [(vec
              (flatten
               (repeat activation-batch-size
                       [0.0066928509242848554 0.01798620996209156 0.04742587317756678
                        0.11920292202211755 0.2689414213699951 0.5 0.7310585786300049
                        0.8807970779778823 0.9525741268224334 0.9820137900379085])))
             (vec
              (flatten
               (repeat activation-batch-size
                       [-0.033240283353950774 -0.07065082485316447 -0.1355299791927364
                        -0.209987170807013 -0.19661193324148185 0.0 0.19661193324148185
                        0.20998717080701323 0.135529979192736 0.07065082485316443])))]
   :tanh [(vec
           (flatten
            (repeat activation-batch-size
                    [-0.9999092042625951 -0.999329299739067 -0.9950547536867305
                     -0.9640275800758169 -0.7615941559557649 0.0 0.7615941559557649
                     0.9640275800758169 0.9950547536867305 0.999329299739067])))
          (vec
           (flatten
            (repeat activation-batch-size
                    [-9.079161547192634E-4 -0.005363802732103666 -0.0295981114963205
                     -0.14130164970632886  -0.41997434161402614 0.0 0.41997434161402614
                     0.14130164970632886 0.0295981114963205 0.005363802732103666])))]})

(defn test-activation-batch
  [backend act-type]
  (let [item-count 10
        batch-size activation-batch-size
        item-range (flatten (repeat batch-size (range (- (/ item-count 2)) (/ item-count 2))))
        ;;sums to zero
        input (nn-backend/array backend item-range batch-size)
        layer (layers/activation backend item-count act-type)
        layer (cp/setup layer batch-size)
        layer (cp/calc layer input)
        output (cp/output layer)
        output-data (nn-backend/to-double-array backend output)]
    (is (utils/about-there? output-data (first (activation-batch-answers act-type)) 1e-3))
    (let [output-gradient (nn-backend/array backend item-range
                                         batch-size)
          layer (cp/backward layer input output-gradient)
          input-gradient (cp/input-gradient layer)
          input-grad-data (nn-backend/to-double-array backend input-gradient)]
      (is (utils/about-there? input-grad-data
                              (second (activation-batch-answers act-type)) 1e-3)))))


(defn softmax
  [backend]
  (let [input (nn-backend/array backend (vec (take 10 (flatten (repeat [1 2 3 4])))))
        layer (layers/softmax backend 10)
        layer (cp/setup layer 1)
        layer (cp/calc layer input)
        output (cp/output layer)
        output-data (nn-backend/to-double-array backend output)]
    (is (utils/about-there? [0.015127670383492609,0.041121271510366035,0.11177920510975863
                             ,0.30384738204945333,0.015127670383492609,0.041121271510366035
                             ,0.11177920510975863,0.30384738204945333,0.015127670383492609
                             ,0.041121271510366035] output-data))
    (let [output-gradient (nn-backend/array backend (repeat 10 1))
          layer (cp/backward layer input output-gradient)
          input-gradient (nn-backend/to-double-array backend (cp/input-gradient layer))]
      (is (= (map double (repeat 10 1))
             (seq input-gradient))))))


(defn softmax-batch
  [backend]
  (let [batch-count 10
        input (nn-backend/array backend (vec (flatten (repeat batch-count
                                                       (take 10 (flatten (repeat [1 2 3 4]))))))
                             batch-count)
        layer (layers/softmax backend 10)
        layer (cp/setup layer batch-count)
        layer (cp/calc layer input)
        output (cp/output layer)
        output-data (nn-backend/to-double-array backend output)]
    (is (utils/about-there?
         (flatten (repeat batch-count
                          [0.015127670383492609,0.041121271510366035,0.11177920510975863
                           ,0.30384738204945333,0.015127670383492609,0.041121271510366035
                           ,0.11177920510975863,0.30384738204945333,0.015127670383492609
                           ,0.041121271510366035])) output-data))
    (let [output-gradient (nn-backend/array backend (repeat (* batch-count 10) 1))
          layer (cp/backward layer input output-gradient)
          input-gradient (nn-backend/to-double-array backend (cp/input-gradient layer))]
      (is (= (map double (repeat (* 10 batch-count) 1))
             (seq input-gradient))))))



(defn softmax-batch-channels
  [backend]
  (let [batch-count 10
        channels 4
        n-input-pixels 10
        input (nn-backend/array backend
                                (vec (repeat batch-count
                                             (take n-input-pixels
                                                   (repeat [1 2 3 4]))))
                             batch-count)
        layer (layers/softmax backend (* channels n-input-pixels) :channels channels)
        layer (cp/setup layer batch-count)
        layer (cp/calc layer input)
        output (cp/output layer)
        output-data (nn-backend/to-double-array backend output)]
    (is (utils/about-there?
         (flatten (repeat batch-count
                          (take n-input-pixels
                                (repeat [0.03205860328008499
                                         0.08714431874203257
                                         0.23688281808991013
                                         0.6439142598879724]))))
         output-data
         1e-4))
    (let [output-gradient (nn-backend/array backend
                                            (repeat (* batch-count
                                                       channels n-input-pixels) 1))
          layer (cp/backward layer input output-gradient)
          input-gradient (nn-backend/to-double-array backend (cp/input-gradient layer))]
      (is (= (map double (repeat (* channels n-input-pixels batch-count) 1))
             (seq input-gradient))))))



(defn create-conv-layer
  [backend input-dim num-channels k-dim pad stride n-kernels]
  (let [conv-config (conv/create-conv-layer-config input-dim input-dim
                                                   k-dim k-dim
                                                   pad pad
                                                   stride stride
                                                   num-channels
                                                   n-kernels)
        weights (nn-backend/array backend (map #(repeat (* k-dim k-dim num-channels) %)
                                        (range 1 (+ n-kernels 1))))
        bias (nn-backend/array backend (vec (repeat n-kernels 1)))]
    (layers/->Convolutional backend weights bias conv-config nil)))


(defn basic-conv-layer
  [backend]
  (let [batch-size 10
        channel-count 4
        conv-layer (create-conv-layer backend 3 1 2 0 1 channel-count)
        input (nn-backend/array backend (repeat batch-size (range 1 10)) batch-size)
        output-gradient (nn-backend/array backend (flatten
                                            (repeat (* 4 batch-size) [1 1 1 1])) batch-size)
        conv-layer (cp/setup conv-layer batch-size)
        conv-layer (cp/forward conv-layer input)
        conv-layer (cp/backward conv-layer input output-gradient)
        input-gradient (cp/input-gradient conv-layer)]
    (is (= (flatten (repeat batch-size [13.0 17.0 25.0 29.0 25.0 33.0 49.0 57.0
                                        37.0 49.0 73.0 85.0 49.0 65.0 97.0 113.0]))
           (seq (nn-backend/to-double-array backend (cp/output conv-layer)))))
    (is (= (map double (flatten (repeat 4 [120.0 160.0 240.0 280.0])))
           (m/eseq (nn-backend/to-double-array backend (:weight-gradient conv-layer)))))
    (is (= (map double (repeat 4 (* 4 batch-size)))
           (m/eseq (nn-backend/to-double-array backend (:bias-gradient conv-layer)))))
    (is (= (flatten (repeat batch-size (map #(double (* 10 %)) [1 2 1 2 4 2 1 2 1])))
           (m/eseq (nn-backend/to-double-array backend input-gradient))))))


(defn pool-layer-basic
  [backend]
  (let [batch-size 10
        pool-layer-config (conv/create-conv-layer-config 2 2 2 2 0 0 1 1 4)
        pool-layer (layers/->Pooling backend pool-layer-config)
        pool-layer (cp/setup pool-layer batch-size)
        input (nn-backend/array backend (flatten (repeat batch-size (range 1 17))) batch-size)
        output-gradient (nn-backend/array backend (flatten (repeat batch-size [1 2 3 4]))
                                          batch-size)
        pool-layer (cp/forward pool-layer input)
        pool-layer (cp/backward pool-layer input output-gradient)
        output (cp/output pool-layer)
        input-gradient (cp/input-gradient pool-layer)]
    (is (= (map double (flatten (repeat batch-size [4 8 12 16])))
           (m/eseq (nn-backend/to-double-array backend output))))
    (is (= (map double (flatten (repeat batch-size (map #(vector 0 0 0 %) (range 1 5)))))
           (m/eseq (nn-backend/to-double-array backend input-gradient))))
    (let [input (nn-backend/array backend (repeat batch-size (range 16 0 -1)) batch-size)
          output-gradient (nn-backend/array backend (flatten (repeat batch-size  [1 2 3 4])) batch-size)
          pool-layer (cp/forward pool-layer input)
          pool-layer (cp/backward pool-layer input output-gradient)
          input-gradient (cp/input-gradient pool-layer)]
      (is (= (map double (flatten (repeat batch-size [16 12 8 4])))
             (m/eseq (nn-backend/to-double-array backend (cp/output pool-layer)))))
      (is (= (map double (flatten (repeat batch-size (map #(vector % 0 0 0) (range 1 5)))))
             (m/eseq (nn-backend/to-double-array backend input-gradient)))))))


(defn count-zeros
  [item-seq]
  (count (filter #(= 0.0 (double %)) item-seq)))


(defn dropout-bernoulli
  [backend]
  (let [batch-size 5
        item-count 20
        input (nn-backend/array backend (repeat (* batch-size item-count) 1.0) batch-size)
        output-gradient (nn-backend/array backend (repeat (* batch-size item-count) 2.0)
                                          batch-size)
        dropout-layer (layers/bernoulli-dropout backend item-count 0.8)
        dropout-layer (cp/setup dropout-layer batch-size)
        repeat-count 30
        answer-seq
        (doall
         (for [iter (range repeat-count)]
           (let [dropout-layer (cp/prepare-forward dropout-layer)
                 dropout-layer (cp/forward dropout-layer input)
                 dropout-layer (cp/backward dropout-layer input output-gradient)
                 output (seq (nn-backend/to-double-array backend (cp/output dropout-layer)))
                 input-gradient (seq
                                 (nn-backend/to-double-array backend
                                                          (cp/input-gradient dropout-layer)))]
             [(m/esum output) (count-zeros output)
              (m/esum input-gradient) (count-zeros input-gradient)])))
        final-aggregate  (reduce m/add answer-seq)
        final-answer (m/div final-aggregate repeat-count)
        total-elem-count (double (* item-count batch-size))]
    ;;zero count should be identical
    (is (= (final-answer 1) (final-answer 3)))
    (is (utils/about-there? (final-answer 0) total-elem-count 3))
    (is (utils/about-there? (final-answer 2) (* 2.0 total-elem-count) 5))))


(defn dropout-gaussian
  [backend]
  (let [batch-size 5
        item-count 100
        input (nn-backend/array backend (repeat (* batch-size item-count) 1.0) batch-size)
        output-gradient (nn-backend/array backend (repeat (* batch-size item-count) 2.0)
                                          batch-size)
        dropout-layer (layers/gaussian-dropout backend item-count 0.5)
        dropout-layer (cp/setup dropout-layer batch-size)
        dropout-layer (cp/prepare-forward dropout-layer)
        repeat-count 30
        answer-seq
        (doall
         (for [iter (range repeat-count)]
           (let [dropout-layer (cp/prepare-forward dropout-layer)
                 dropout-layer (cp/forward dropout-layer input)
                 dropout-layer (cp/backward dropout-layer input output-gradient)
                 output (seq (nn-backend/to-double-array backend (cp/output dropout-layer)))
                 input-gradient (seq (nn-backend/to-double-array backend (cp/input-gradient
                                                                   dropout-layer)))]
             [(m/esum output) (m/esum input-gradient)])))
        final-aggregate  (reduce m/add answer-seq)
        final-answer (m/div final-aggregate repeat-count)
        total-elem-count (double (* item-count batch-size))]
    (is (utils/about-there? (final-answer 0) total-elem-count 10))
    (is (utils/about-there? (final-answer 1) (* 2.0 total-elem-count) 20))))


(defn split-basic
  [backend]
  (let [item-count 1000
        items-per-batch 5
        ;;sums to zero
        input (nn-backend/array backend (flatten (repeat (* items-per-batch
                                                         (/ item-count 2)) [-1 1]))
                             items-per-batch)
        layer (layers/split backend [(layers/activation backend item-count :relu)
                                     (layers/activation backend item-count :relu)]
                            item-count)
        layer (cp/setup layer items-per-batch)
        layer (cp/multi-forward layer [input])
        output (cp/multi-output layer)
        _ (is (= 2 (count output)))
        output-data (mapv #(nn-backend/to-double-array backend %) output)]

    (is (every? #(= (double (* items-per-batch
                               (/ item-count 2)))
                    %)
                (map m/esum output-data)))

    (let [output-gradient (nn-backend/array backend (repeat (* items-per-batch item-count) 1)
                                         items-per-batch)
          output-gradient [output-gradient output-gradient]
          layer (cp/multi-backward layer [input] output-gradient)
          input-gradient (first (cp/multi-input-gradient layer))
          input-grad-data (nn-backend/to-double-array backend input-gradient)]
      (is (= (double (* items-per-batch item-count))
             (m/esum input-grad-data))))))


(defn batch-normalization
  [backend]
  (let [batch-size 20
        input-size 20
        input-data-vector-fn (fn []
                               (m/transpose
                                (repeatedly input-size
                                            #(-> (repeatedly batch-size cu/rand-gaussian)
                                                 double-array
                                                 (cu/ensure-gaussian! 5 20)))))
        input-data-vector (input-data-vector-fn)
        layer (cp/setup
               (layers/batch-normalization backend input-size 0.8)
               batch-size)
        input (nn-backend/array backend input-data-vector batch-size)
        layer (cp/forward layer input)
        output (cp/output layer)
        double-output (nn-backend/to-double-array backend output)
        output-batches (mapv vec (partition input-size (seq double-output)))
        output-stats (mapv cu/calc-mean-variance (m/transpose output-batches))
        input-stats (mapv cu/calc-mean-variance (m/transpose input-data-vector))
        layer-means (vec (nn-backend/to-double-array backend (:batch-means layer)))
        layer-variances (vec (nn-backend/to-double-array backend (:batch-variances layer)))
        layer-stats (vec (map (fn [mean variance]
                                {:mean mean
                                 :variance variance})
                              layer-means layer-variances))]
    (doseq [output-idx (range (count output-stats))]
      (let [{:keys [mean variance]} (output-stats output-idx)]
        (is (utils/about-there? mean 0.0)
            (format "Output mean incorrect at index %s" output-idx))
        (is (utils/about-there? variance 1.0 1e-3)
            (format "Output variance incorrect at index %s" output-idx))))
    (dotimes [iter 5]
     (let [input-data-vector (input-data-vector-fn)
           new-input (nn-backend/array backend input-data-vector batch-size)
           layer (cp/forward layer new-input)
           output (cp/output layer)
           double-output (nn-backend/to-double-array backend output)
           output-batches (mapv vec (partition input-size (seq double-output)))
           output-stats (mapv cu/calc-mean-variance (m/transpose output-batches))]
       (doseq [output-idx (range (count output-stats))]
         (let [{:keys [mean variance]} (output-stats output-idx)]
           (is (utils/about-there? mean 0.0)
               (format "Output mean incorrect at index %s" output-idx))
           (is (utils/about-there? variance 1.0 1e-3)
               (format "Output variance incorrect at index %s" output-idx))))))
    (let [running-means (nn-backend/to-double-array backend (:running-means layer))
          running-inv-vars (nn-backend/to-double-array backend (:running-variances layer))]
      (is (utils/about-there? 5.0 (/ (m/esum running-means)
                                     input-size)))
      ;;The running variances uses a population calculation for variances
      ;;instead of a specific calculation for variance meaning
      ;;you divide by n-1 instead of n.
      (is (utils/about-there? 21.05 (/ (m/esum running-inv-vars)
                                      input-size)
                              1e-2)))))

(defn do-lrn-forward
  [backend num-input-channels lrn-n]
  (resource/with-resource-context
   (let [batch-size 2
         input-dim 2
         input-num-pixels (* input-dim input-dim)
         n-input (* num-input-channels input-num-pixels)
         input-data (flatten (repeat batch-size (range n-input)))
         input (math/with-tensor
                 (nn-backend/array backend input-data batch-size)
                 (math/map->Tensor {:batch-size batch-size
                                    :channel-count num-input-channels
                                    :width input-dim
                                    :height input-dim} ))
         layer (cp/setup (layers/local-response-normalization
                          backend
                          input-dim input-dim num-input-channels
                          :k 1 :n lrn-n :alpha 1.0 :beta 1.0)
                         batch-size)
         layer (cp/forward layer input)
         output (nn-backend/to-double-array backend (cp/output layer))
         output-gradient (math/with-tensor
                           (nn-backend/array backend (repeat (* batch-size n-input) 1.0))
                           (math/map->Tensor {:batch-size batch-size
                                              :channel-count num-input-channels
                                              :width input-dim
                                              :height input-dim}))
         layer (cp/backward layer input output-gradient)
         input-gradient (nn-backend/to-double-array backend (cp/input-gradient layer))]
     {:input-data input-data
      :output (vec output)
      :input-gradient (vec input-gradient)})))

(defn lrn-forward
  [backend]
  (let [lrn-data (do-lrn-forward backend 3 1)]
    (is (m/equals (:output lrn-data)
                  (mapv #(/ (double %) (+ 1 (* % %))) (:input-data lrn-data))
                  1e-4)))
  (let [lrn-data (do-lrn-forward backend 3 2)]
    (is (m/equals (:output lrn-data)
                  (mapv double
                        (flatten
                         (repeat 2
                                 [0.0 0.07142857142857142 0.09523809523809523 0.1
                                  0.0975609756097561 0.09259259259259259 0.08695652173913043
                                  0.08139534883720931 0.24242424242424243 0.21686746987951808
                                  0.19607843137254902 0.17886178861788618])))
                  1e-4)))
  (let [lrn-data (do-lrn-forward backend 3 3)]
    (is (m/equals (:output lrn-data)
                  (mapv double
                        (flatten
                         (repeat 2
                                 [0.0 0.10344827586206898 0.13953488372093023
                                  0.14754098360655737 0.14457831325301207 0.13636363636363638
                                  0.1258741258741259 0.11538461538461539 0.28915662650602414
                                  0.24770642201834867 0.21582733812949642
                                  0.19075144508670522 ])))
                  1e-4))))
