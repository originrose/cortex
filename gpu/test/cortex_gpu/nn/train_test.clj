(ns cortex-gpu.nn.train-test
  (:require [clojure.test :refer :all]
            [cortex-gpu.nn.train :as train]
            [cortex-gpu.nn.cudnn :as cudnn]
            [cortex-gpu.nn.layers :as layers]
            [cortex-gpu.resource :as resource]
            [cortex-gpu.nn.description :as gpu-desc]
            [cortex.nn.description :as desc]
            [cortex.nn.protocols :as cp]
            [cortex-gpu.test-framework :refer [def-double-float-test] :as framework]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.protocols :as mp]
            [cortex-datasets.mnist :as mnist]
            [cortex-gpu.nn.batch :as batch]
            [cortex-gpu.cuda :as cuda]
            [cortex.optimise :as opt]
            [cortex.nn.core :as core]
            [cortex.nn.network :as net]
            [cortex.nn.backends :as b]
            [cortex.nn.description :as desc]
            [cortex-gpu.nn.description :as gpu-desc]
            [cortex-gpu.util :as util]
            [cortex.optimise :as opt]))


(use-fixtures :once framework/with-contexts)
(use-fixtures :each framework/with-resource-context)


(def-double-float-test loss-gradient
  (let [num-items 100
        output (cudnn/array (range num-items))
        answer (cudnn/array (repeat num-items 1))
        output-gradient (cudnn/new-array [(cudnn/ecount output)])
        alpha 5]
    (cudnn/loss-gradient alpha output answer output-gradient)
    (is (framework/about-there? (m/mul (m/sub (range num-items) (repeat num-items 1)) alpha)
                                (cudnn/to-double-array output-gradient)))))


(def-double-float-test many-operations
  (let [many-data [[1 2 3] (vec (repeat 20 5)) (vec (range -100 0))]
        packed-data (vec (flatten many-data))
        many-cudnn (mapv #(cudnn/array %) many-data)
        packed-cudnn (cudnn/new-array [(util/many-ecount many-cudnn)])]

    (util/assign-many->packed many-cudnn packed-cudnn)
    (is (m/equals packed-data (seq (cudnn/to-double-array packed-cudnn))))
    (cudnn/zero! packed-cudnn)
    (util/assign-packed->many packed-cudnn many-cudnn)
    (let [many-test (mapv (fn [item]
                            (vec (cudnn/to-double-array item)))
                          many-cudnn)]
      (is (= [(mapv double [0 0 0]) (vec (repeat 20 0.0)) (vec (repeat 100 0.0))]
             many-test)))))


(def-double-float-test train-step
  (let [num-batch-items 10
        weights (cudnn/array [[1 2] [3 4]])
        bias (cudnn/array [0 10])
        input (cudnn/array (flatten (repeat num-batch-items [1 2])) num-batch-items)
        layer (layers/->Linear weights bias nil)
        layer (layers/setup layer num-batch-items)
        train-config {:network layer :loss-fn (opt/mse-loss)}
        layer (:network
               (train/train-step train-config input [(cudnn/array
                                                       (flatten
                                                        (repeat num-batch-items [4 19])))]))
        output (cp/output layer)
        output-data (cudnn/to-double-array output)
        weight-gradient (vec (cudnn/to-double-array (:weight-gradient layer)))
        bias-gradient (vec (cudnn/to-double-array (:bias-gradient layer)))
        input-gradient (vec (cudnn/to-double-array (:input-gradient layer)))]
    (is (= (map double (flatten (repeat num-batch-items [5 21])))
           (m/eseq output-data)))
    (is (m/equals (mapv #(* % num-batch-items) [1 2 2 4]) weight-gradient))
    (is (m/equals (mapv #(* % num-batch-items) [1 2]) bias-gradient))
    (is (m/equals (flatten (repeat num-batch-items [7 10])) input-gradient))))


(deftest optimise
  (let [num-batch-items 10
        weights (cudnn/array [[1 2] [3 4]])
        bias (cudnn/array [0 10])
        input (cudnn/array (flatten (repeat num-batch-items [1 2])) num-batch-items)
        layer (layers/->Linear weights bias nil)
        layer (layers/setup layer num-batch-items)
        train-config (-> {:network layer
                          :loss-fn (opt/mse-loss)
                          :optimiser (opt/adadelta-optimiser)
                          :batch-size num-batch-items}
                         (train/train-step input [(cudnn/array
                                                    (flatten (repeat num-batch-items [4 19])))])
                         train/optimise)
        layer (:network train-config)
        optimiser (:optimiser train-config)]
    (is (framework/about-there? (seq (cudnn/to-double-array weights))
                                [0.9955279087656892 1.9955278752252983
                                 2.9955278752252985 3.995527866840083]))
    (is (framework/about-there? (seq (cudnn/to-double-array bias))
                                [-0.004472091234310839 9.995527875225298]))
    (is (framework/about-there? (seq (cudnn/to-double-array (:grad-accum optimiser)))
                                [0.05 0.2 0.2 0.8 0.05 0.2]))
    (is (framework/about-there? (seq (cudnn/to-double-array (:dx-accum optimiser)))
                                [9.999800003999923E-7 9.99995000025E-7
                                 9.99995000025E-7 9.999987500015622E-7
                                 9.999800003999923E-7 9.99995000025E-7]))
    (is (framework/about-there? (seq (cudnn/to-double-array (:weight-gradient layer)))
                                [0 0 0 0]))
    (is (framework/about-there? (seq (cudnn/to-double-array (:bias-gradient layer)))
                                [0 0]))))




; Data from: Dominick Salvator and Derrick Reagle
; Shaum's Outline of Theory and Problems of Statistics and Economics
; 2nd edition,  McGraw-Hill, 2002, pg 157

; Predict corn yield from fertilizer and insecticide inputs
; [corn, fertilizer, insecticide]
(def CORN-DATA
  [[6  4]
   [10  4]
   [12  5]
   [14  7]
   [16  9]
   [18 12]
   [22 14]
   [24 20]
   [26 21]
   [32 24]])


(def CORN-LABELS
  [[40] [44] [46] [48] [52] [58] [60] [68] [74] [80]])

(def-double-float-test corn
  (with-bindings {#'train/*train-epoch-reporter* nil}
    (let [net (layers/linear 2 1)
          n-epochs 5000
          loss (opt/mse-loss)
          optimizer (opt/adadelta-optimiser)
          net (train/train net optimizer loss CORN-DATA CORN-LABELS 1 n-epochs)
          results (train/run net CORN-DATA)
          mse (train/mse results CORN-LABELS)]
      (is (< mse 25)))))


(def training-data (future (mnist/training-data)))
(def training-labels (future (mnist/training-labels)))
(def test-data (future (mnist/test-data)))
(def test-labels (future (mnist/test-labels)))

(def basic-network-description
  [(desc/input 28 28 1)
   (desc/convolutional 5 0 1 20)
   (desc/max-pooling 2 0 2)
   (desc/convolutional 5 0 1 50)
   (desc/max-pooling 2 0 2)
   (desc/linear->relu 500)
   (desc/linear->softmax 10)])

(defn maybe-takev
  [item-count coll]
  (if item-count
    (vec (take item-count coll))
    coll))

(defn default-mnist-labels
  []
  {:training #(maybe-takev % @training-labels)
   :test #(maybe-takev % @test-labels)})

(defn train-mnist-network
  "Train an mnist network.  This function is somewhat abstracted so that
you can train a network that is either straight or branched.
All labels are expected to be futures."
  [{:keys [max-sample-count loss-fn label-fns network-description]
    :or {loss-fn (opt/->SoftmaxCrossEntropyLoss)
         label-fns (default-mnist-labels)
         network-description basic-network-description}}]
  (let [input-count (count (first @training-data))
        batch-size 10
        network (gpu-desc/build-and-create-network network-description)
        epoch-count 4
        take-max-samples (if max-sample-count
                           #(maybe-takev max-sample-count %)
                           identity)
        training-labels-fn (get label-fns :training)
        test-labels-fn (get label-fns :test)]
    (time (train/train
           network
           (opt/adadelta-optimiser)
           loss-fn
           (take-max-samples @training-data)
           (training-labels-fn max-sample-count)
           batch-size
           epoch-count
           (take-max-samples @test-data)
           (test-labels-fn max-sample-count)))))


(defn train-mnist
  ([{:keys [max-sample-count loss-fn label-fns]
     :or {label-fns (default-mnist-labels)} :as options}]
   (let [network (train-mnist-network options)
         test-labels-fn (get label-fns :test)]
     (println (format "network results: %s" (train/evaluate-softmax
                                             network
                                             @test-data
                                             (test-labels-fn nil))))))
  ;;All options have sane defaults
  ([] (train-mnist {})))




(defn full-cpu-mnist
  []
  (let [batch-size 10
        n-epochs 4
        network (desc/build-and-create-network basic-network-description)
        optimizer (opt/adadelta-optimiser)
        loss-fn (opt/->SoftmaxCrossEntropyLoss)
        _ (println "training")
        network (net/train network optimizer loss-fn
                           (mapv b/array @training-data)
                           (mapv b/array @training-labels)
                           batch-size n-epochs
                           (mapv b/array @test-data)
                           (mapv b/array @test-labels))]
    (println "evaluating")
    (println (format "network results: %f"
                     (net/evaluate-softmax network
                                           (mapv b/array @test-data)
                                           (mapv b/array @test-labels))))))

(def-double-float-test partial-mnist
  (train-mnist {:max-sample-count 100}))

(def SMALL-NUM 1e-30)

(defn ce-loss [ v target]
  (let [a (m/mul (m/negate target) (m/log (m/add SMALL-NUM v)))
        b (m/mul (m/sub 1.0 target) (m/log (m/sub (+ 1.0 (double SMALL-NUM)) v)))
        c (m/esum (m/sub a b))]
    c))


(defn full-mnist-d
  []
  (framework/with-contexts
    (fn [] (train-mnist))))

(defn full-mnist-f
  []
  (with-bindings {#'cudnn/*cudnn-datatype* (float 0.0)}
   (framework/with-contexts
     (fn [] (train-mnist)))))


(def-double-float-test layer->description
  (let [network (train-mnist-network {:max-sample-count 100})
        results (train/evaluate-softmax network
                                        (mapv b/array @test-data)
                                        (mapv b/array @test-labels))
        network-desc (gpu-desc/network->description network)
        new-network (gpu-desc/build-and-create-network network-desc)
        new-results (train/evaluate-softmax new-network
                                            (mapv b/array @test-data)
                                            (mapv b/array @test-labels))]
    (is (m/equals results new-results))))



(def adam-answers
  [[0.999000000005, 1.9990000000025, 2.9990000000016668, 3.99900000000125]
   [0.9980000262138343, 1.9980000130723587, 2.998000008707231, 3.998000006527546]
   [0.9970000960651408, 1.9970000478972731, 2.9970000319014805, 3.9970000239148513]
   [0.9960002269257634, 1.9960001131228113, 2.9960000753397424, 3.9960000564765217]
   [0.995000436052392, 1.9950002173334274, 2.995000144735273, 3.995000108493863]
   [0.9940007405541528, 1.9940003690339771, 2.994000245746984, 3.9940001842069432]
   [0.9930011573564278, 1.9930005766318695, 2.9930003839675967, 3.993000287805732]
   [0.9920017031661642, 1.9920008484199827, 2.992000564912314, 3.992000423421624]
   [0.9910023944389119, 1.991001192560463, 2.9910007940080785, 3.9910005951194076]
   [0.9900032473478027, 1.9900016170695067, 2.9900010765834972, 3.990000806889731]
   [0.9890042777546556, 1.9890021298032192, 2.9890014178594777, 3.9890010626421075]
   [0.9880055011833645, 1.9880027384446224, 2.9880018229406398, 3.988001366198499]
   [0.9870069327956914, 1.987003450491873, 2.98700229680753, 3.9870017212875037]
   [0.9860085873695607, 1.986004273247733, 2.9860028443096756, 3.9860021315391725]
   [0.9850104792799151, 1.9850052138103236, 2.9850034701594894, 3.9850026004804673]
   [0.9840126224821667, 1.984006279065173, 2.984004178927043, 3.984003131531364]
   [0.9830150304982433, 1.9830074756785594, 2.9830049750356973, 3.9830037280016044]
   [0.9820177164052082, 1.9820088100921336, 2.982005862758586, 3.9820043930880837]
   [0.9810206928263993, 1.9810102885187928, 2.981006846215933, 3.9810051298728633]])


(def all-adam-output
  [[0.999000000005, 1.9990000000025, 2.9990000000016668, 3.99900000000125]
   [0.19999999999999996, 0.3999999999999999, 0.5999999999999999, 0.7999999999999998]
   [0.0040000000000000036, 0.016000000000000014, 0.03600000000000003, 0.06400000000000006]
   0.9
   0.999
   [0.9980000262138343, 1.9980000130723587, 2.998000008707231, 3.998000006527546]
   [0.3798000000009999, 0.7598000000004999, 1.1398000000003332, 1.5198000000002496]
   [0.007988004000039968, 0.03196800400004001, 0.07194000400004005, 0.12790400400004012]
   0.81
   0.998001
   [0.9970000960651408, 1.9970000478972731, 2.9970000319014805, 3.9970000239148513]
   [0.5414200052436667, 1.0834200026149214, 1.625420001741746, 2.1674200013057336]
   [0.011964032205331186, 0.04790405220498857, 0.10782008020487427, 0.19171211620481715]
   0.7290000000000001
   0.997002999
   [0.9960002269257634, 1.9960001131228113, 2.9960000753397424, 3.9960000564765217]
   [0.6866780239323282, 1.3744780119328839, 2.0622780079478673, 2.7500780059581302]
   [0.015928104939341457, 0.06380818491799044, 0.14364029688953933, 0.2554244408533137]
   0.6561000000000001
   0.996005996001
   [0.995000436052392, 1.9950002173334274, 2.995000144735273, 3.995000108493863]
   [0.817210266924248, 1.6362302333641578, 2.455250222221029, 3.2742702166576216]
   [0.01988024264254681, 0.07968044253941756, 0.1794007223983928, 0.3190410822179019]
   0.5904900000000002
   0.995009990004999
   [0.9940007405541528, 1.9940003690339771, 2.994000245746984, 3.9940001842069432]
   [0.9344893274423016, 1.8716072534944275, 2.8087252289459803, 3.745843216690632]
   [0.023820465870882067, 0.09552086556551984, 0.21510142514385167, 0.3825621446031479]
   0.5314410000000002
   0.994014980014994])


(deftest adam-test
  (let [parameters (cudnn/array [1 2 3 4])
        optimizer (opt/adam)]
    (reduce (fn [optimizer item]
              (let [gradient (cudnn/array (map #(* 2.0 %) (cudnn/to-double-array parameters)))
                    optimizer (cortex-gpu.optimise/compute-parameters! optimizer [gradient] [parameters] 1)
                    item (m/eseq item)
                    guess (m/eseq (cudnn/to-double-array parameters))]
                (comment (println guess)
                         (println (m/eseq (cudnn/to-double-array (:m optimizer))))
                         (println (m/eseq (cudnn/to-double-array (:v optimizer))))
                         (println (m/eseq (:pow-beta1-t optimizer)))
                         (println (m/eseq (:pow-beta2-t optimizer))))
                (resource/release (:ptr gradient))
                (resource/release (:tensor gradient))
                (is (m/equals guess item))
                optimizer))
            optimizer
            adam-answers)))

(def-double-float-test split-train-test
  (testing "Test an autoencoder that trains some nist for decode and softmax final stages"
    (let [max-sample-count 100
          network-description
          [(desc/input 28 28 1)
           (desc/dropout 0.8)
           (desc/linear->logistic 144 :l2-max-constraint 2.0)
           (desc/dropout 0.5)
           ;;Note carefully the order of the leaves of the network.  There
           ;;is currently an implicit dependency here on that order, the order
           ;;of the loss functions and the order of the training and test
           ;;labels which is probably an argument to specify all of that
           ;;in the network description
           (desc/split [[(desc/linear 784)] [(desc/linear->softmax 10)]])]

          ;;The order of the labels has to match the order of the leaves of the network.
          ;;in this case we have a decoder followed by a classifier
          make-split-network-label-fn (fn [data labels]
                                        (fn [sample-count]
                                          [(maybe-takev sample-count @data)
                                           (maybe-takev sample-count @labels)]))
          ;;The order of the loss functions has to match the order of the leaves.
          ;;MSELoss (for the decoder) and softmax loss (for the classifier)
          loss-fn [(opt/->MSELoss) (opt/->SoftmaxCrossEntropyLoss)]
          ;;The order of the loss functions has to match the order of the leaves.
          ;;MSELoss (for the decoder) and softmax loss (for the classifier)
          labels {:training (make-split-network-label-fn training-data training-labels)
                  :test (make-split-network-label-fn test-data test-labels)}]
      (train-mnist-network {:max-sample-count max-sample-count
                            :network-description network-description
                            :loss-fn loss-fn
                            :label-fns labels}))))
