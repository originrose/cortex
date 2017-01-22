(ns cortex.verify.nn.layers
  "Verify that layers do actually produce their advertised results."
  (:require [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
            [cortex.nn.traverse :as traverse]
            [cortex.nn.execute :as execute]
            [clojure.test :refer :all]
            [clojure.core.matrix :as m]
            [cortex.verify.utils :as utils]
            [cortex.gaussian :as cu]
            [think.resource.core :as resource]
            [cortex.nn.protocols :as cp]))

(defn bind-test-network
  [context network batch-size stream->size-map test-layer-id]
  (let [test-layer-id :test
        input-bindings [(traverse/->input-binding :input :data)]
        output-bindings [(traverse/->output-binding test-layer-id :stream :labels)]]
    (as-> network network
      (flatten network)
      (vec network)
      (assoc-in network [0 :id] :input)
      (assoc-in network [1 :id] test-layer-id)
      (network/build-network network)
      (traverse/bind-input-bindings network input-bindings)
      (traverse/bind-output-bindings network output-bindings)
      (traverse/network->training-traversal network stream->size-map :keep-non-trainable? true)
      (assoc network :batch-size batch-size)
      (cp/bind-to-network context network {}))))


(defn unpack-bound-network
  [context network test-layer-id]
  (let [network (cp/save-to-network context network {:save-gradients? true})
        traversal (get network :traversal)
        test-node (get-in network [:layer-graph :id->node-map test-layer-id])
        parameter-descriptions (layers/get-parameter-descriptions test-node)
        parameters (->> (map (fn [{:keys [key] :as description}]
                               (let [parameter (get test-node key)
                                     buffer (get-in network
                                                    [:layer-graph :buffers
                                                     (get parameter :buffer-id)])]
                                 [key
                                  {:description description
                                   :parameter parameter
                                   :buffer buffer}]))
                             parameter-descriptions)
                        (into {}))
        {:keys [forward buffers]} traversal
        {:keys [incoming id outgoing]} (->> forward
                                            (filter #(= test-layer-id
                                                        (get % :id)))
                                            first)
        incoming-buffers (mapv buffers incoming)
        outgoing-buffers (mapv buffers outgoing)]
    {:network network
     :parameters parameters
     :incoming-buffers incoming-buffers
     :outgoing-buffers outgoing-buffers
     :output (get (first outgoing-buffers) :buffer)
     :input-gradient (get (first incoming-buffers) :gradient)
     :numeric-input-gradient (get (first incoming-buffers) :numeric-gradient)}))


(defn forward-backward-bound-network
  [context network input output-gradient test-layer-id]
  (let [input-stream {:data input}
        output-gradient-stream {test-layer-id output-gradient}]
    (as-> (cp/forward-backward context network input-stream output-gradient-stream) network
      (unpack-bound-network context network test-layer-id))))


(defn forward-backward-test
  "Given a 2 node network (input,layer) run a test that goes forward and backward
for that network."
  [context network batch-size input output-gradient]
  (resource/with-resource-context
    (let [test-layer-id :test
          data-size (/ (m/ecount input)
                       batch-size)
          labels-size (/ (m/ecount output-gradient)
                         batch-size)]
      (as-> (bind-test-network context network batch-size {:data data-size
                                                           :labels labels-size}
                               test-layer-id) network
       (forward-backward-bound-network context network input output-gradient test-layer-id)))))


(defn relu-activation
  [context]
  (let [item-count 10
        ;;sums to zero
        {:keys [output input-gradient]}
        (forward-backward-test context
                               [(layers/input item-count)
                                (layers/relu)]
                               1
                               ;;input sums to zero
                               (flatten (repeat (/ item-count 2) [-1 1]))
                               ;;output-gradient sums to item-count
                               (repeat item-count 1))]
    (is (= (double (/ item-count 2))
           (m/esum output)))
    (is (= (double (/ item-count 2))
           (m/esum input-gradient)))))


(defn relu-activation-batch
  [context]
  (let [item-count 10000
        batch-size 5
        {:keys [output input-gradient]}
        (forward-backward-test context
                               [(layers/input item-count)
                                (layers/relu)]
                               batch-size
                               (flatten (repeat (* batch-size
                                                   (/ item-count 2)) [-1 1]))
                               (repeat (* batch-size item-count) 1))]
    (is (= (double (* batch-size
                      (/ item-count 2)))
           (m/esum output)))
    (is (= (double (* batch-size
                      (/ item-count 2)))
           (m/esum input-gradient)))))


(defn linear
  [context]
  (let [item-count 2
        batch-size 1
        num-output 2
        {:keys [output input-gradient parameters]}
        (forward-backward-test context
                               [(layers/input item-count)
                                (layers/linear num-output
                                               :weights {:buffer [[1 2] [3 4]]}
                                               :bias {:buffer [0 10]})]
                               batch-size
                               [1 2]
                               [1 2])
        weight-gradient (m/eseq (get-in parameters [:weights :buffer :gradient]))
        bias-gradient (m/eseq (get-in parameters [:bias :buffer :gradient]))]
    (is (= (map double [5 21]) (m/eseq output)))
    (is (m/equals [1 2 2 4] weight-gradient))
    (is (m/equals [1 2] bias-gradient))
    (is (m/equals [7 10] input-gradient))))


(defn linear-batch
  [context]
  (let [batch-size 10
        item-count 2
        num-output 2
        {:keys [output input-gradient parameters]}
        (forward-backward-test context
                               [(layers/input item-count)
                                (layers/linear num-output
                                               :weights {:buffer [[1 2] [3 4]]}
                                               :bias {:buffer [0 10]})]
                               batch-size
                               (flatten (repeat batch-size [1 2]))
                               (flatten (repeat batch-size [1 2])))
        weight-gradient (m/eseq (get-in parameters [:weights :buffer :gradient]))
        bias-gradient (m/eseq (get-in parameters [:bias :buffer :gradient]))]
    (is (= (map double (flatten (repeat batch-size [5 21])))
           (m/eseq output)))
    (is (m/equals (mapv #(* % batch-size) [1 2 2 4]) weight-gradient))
    (is (m/equals (mapv #(* % batch-size) [1 2]) bias-gradient))
    (is (m/equals (flatten (repeat batch-size [7 10])) (m/eseq input-gradient)))))


(def activation-answers
  {:logistic [[0.2689414213699951 0.7310585786300049 0.2689414213699951 0.7310585786300049
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
  [context act-type]
  (let [item-count 10
        {:keys [output input-gradient]}
        (forward-backward-test context
                               [(layers/input item-count)
                                {:type act-type}]
                               1
                               (flatten (repeat (/ item-count 2) [-1 1]))
                               (flatten (repeat (/ item-count 2) [-1 1])))]
    (is (utils/about-there? output (first (activation-answers act-type))))
    (is (utils/about-there? input-gradient (second (activation-answers act-type))))))


(def activation-batch-size 5)

(def activation-batch-answers
  {:logistic [(vec
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
  [context act-type]
  (let [item-count 10
        batch-size activation-batch-size
        item-range (flatten (repeat batch-size (range (- (/ item-count 2)) (/ item-count 2))))
        {:keys [output input-gradient]}
        (forward-backward-test context
                               [(layers/input item-count)
                                {:type act-type}]
                               batch-size
                               item-range
                               item-range)]
    (is (utils/about-there? (m/eseq output) (first (activation-batch-answers act-type)) 1e-3))
    (is (utils/about-there? (m/eseq input-gradient) (second (activation-batch-answers act-type)) 1e-3))))


(defn softmax
  [context]
  (let [item-count 10
        batch-size 1
        input (vec (take item-count (flatten (repeat [1 2 3 4]))))
        output-gradient (repeat 10 1)
        {:keys [output input-gradient]}
        (forward-backward-test context
                               [(layers/input item-count)
                                (layers/softmax)]
                               batch-size
                               input
                               output-gradient)]
    (is (utils/about-there? [0.015127670383492609,0.041121271510366035,0.11177920510975863
                             ,0.30384738204945333,0.015127670383492609,0.041121271510366035
                             ,0.11177920510975863,0.30384738204945333,0.015127670383492609
                             ,0.041121271510366035]
                            (m/eseq output)))
    (is (= (map double (repeat 10 1))
           (m/eseq input-gradient)))))


(defn softmax-batch
  [context]
  (let [batch-size 10
        item-count 10
        input (vec (flatten (repeat batch-size
                                    (take item-count (flatten (repeat [1 2 3 4]))))))
        output-gradient (repeat (* batch-size item-count) 1)
        {:keys [output input-gradient]}
        (forward-backward-test context
                               [(layers/input item-count)
                                (layers/softmax)]
                               batch-size
                               input
                               output-gradient)]
    (is (utils/about-there?
         (flatten (repeat batch-size
                          [0.015127670383492609,0.041121271510366035,0.11177920510975863
                           ,0.30384738204945333,0.015127670383492609,0.041121271510366035
                           ,0.11177920510975863,0.30384738204945333,0.015127670383492609
                           ,0.041121271510366035]))
         (m/eseq output)))
    (is (= (map double (repeat (* item-count batch-size) 1))
           (m/eseq input-gradient)))))


(defn softmax-batch-channels
  [context]
  (let [batch-size 10
        channels 4
        n-input-pixels 10
        input (vec (repeat batch-size
                           (take n-input-pixels
                                 (repeat [1 2 3 4]))))
        output-gradient (repeat (* batch-size
                                   channels n-input-pixels) 1)
        {:keys [output input-gradient]}
        (forward-backward-test context
                               [(layers/input (* channels n-input-pixels))
                                (layers/softmax :output-channels channels)]
                               batch-size
                               input
                               output-gradient)]
    (is (utils/about-there?
         (flatten (repeat batch-size
                          (take n-input-pixels
                                (repeat [0.03205860328008499
                                         0.08714431874203257
                                         0.23688281808991013
                                         0.6439142598879724]))))
         (m/eseq output)
         1e-4))
    (is (= (map double (repeat (* channels n-input-pixels batch-size) 1))
           (m/eseq input-gradient)))))


(defn create-conv-layer
  [input-dim num-channels k-dim pad stride n-kernels]
  [(layers/input input-dim input-dim num-channels)
   (layers/convolutional k-dim pad stride n-kernels
                         :weights {:buffer (map #(repeat (* k-dim k-dim num-channels) %)
                                                (range 1 (+ n-kernels 1)))}
                         :bias {:buffer (vec (repeat n-kernels 1))})])


(defn basic-conv-layer
  [context]
  (let [batch-size 10
        channel-count 4
        input (repeat batch-size (range 1 10))
        output-gradient (flatten
                         (repeat (* 4 batch-size) [1 1 1 1]))
        {:keys [output input-gradient parameters]}
        (forward-backward-test context
                               (create-conv-layer 3 1 2 0 1 channel-count)
                               batch-size
                               input
                               output-gradient)]
    (is (= (flatten (repeat batch-size [13.0 17.0 25.0 29.0 25.0 33.0 49.0 57.0
                                        37.0 49.0 73.0 85.0 49.0 65.0 97.0 113.0]))
           (m/eseq output)))
    (is (= (map double (flatten (repeat 4 [120.0 160.0 240.0 280.0])))
           (m/eseq (get-in parameters [:weights :buffer :gradient]))))
    (is (= (map double (repeat 4 (* 4 batch-size)))
           (m/eseq (get-in parameters [:bias :buffer :gradient]))))
    (is (= (flatten (repeat batch-size (map #(double (* 10 %)) [1 2 1 2 4 2 1 2 1])))
           (m/eseq input-gradient)))))



(defn pool-layer-basic
  [context]
  (let [batch-size 10
        num-input 16
        input (flatten (repeat batch-size (map inc (range num-input))))
        output-gradient (flatten (repeat batch-size [1 2 3 4]))
        {:keys [output input-gradient]}
        (forward-backward-test context
                               [(layers/input 2 2 4)
                                (layers/max-pooling 2 0 1)]
                               batch-size
                               input
                               output-gradient)]
    (is (= (map double (flatten (repeat batch-size [4 8 12 16])))
           (m/eseq output)))
    (is (= (map double (flatten (repeat batch-size (map #(vector 0 0 0 %) (range 1 5)))))
           (m/eseq input-gradient)))
    (let [input (repeat batch-size (range 16 0 -1))
          output-gradient (flatten (repeat batch-size  [1 2 3 4]))
          {:keys [output input-gradient]}
          (forward-backward-test context
                                 [(layers/input 2 2 4)
                                  (layers/max-pooling 2 0 1)]
                                 batch-size
                                 input
                                 output-gradient)]
      (is (= (map double (flatten (repeat batch-size [16 12 8 4])))
             (m/eseq output)))
      (is (= (map double (flatten (repeat batch-size (map #(vector % 0 0 0) (range 1 5)))))
             (m/eseq input-gradient))))))


(defn count-zeros
  [item-seq]
  (count (filter #(= 0.0 (double %)) item-seq)))


(defn dropout-bernoulli
  [context]
  (resource/with-resource-context
   (let [batch-size 5
         item-count 20
         repeat-count 30
         input (repeat (* batch-size item-count) 1.0)
         output-gradient (repeat (* batch-size item-count) 2.0)
         test-layer-id :test
         dropout-network (bind-test-network context
                                            [(layers/input item-count)
                                             (layers/dropout 0.8)]
                                            batch-size
                                            {:data item-count
                                             :labels item-count}
                                            test-layer-id)
         answer-seq
         (doall
          (for [iter (range repeat-count)]
            (let [{:keys [output input-gradient]}
                  (forward-backward-bound-network context dropout-network
                                                 input output-gradient
                                                 test-layer-id)
                  output (m/eseq output)
                  input-gradient (m/eseq input-gradient)]
              [(m/esum output) (count-zeros output)
               (m/esum input-gradient) (count-zeros input-gradient)])))
         final-aggregate  (reduce m/add answer-seq)
         final-answer (m/div final-aggregate repeat-count)
         total-elem-count (double (* item-count batch-size))]
     ;;zero count should be identical
     (is (= (final-answer 1) (final-answer 3)))
     (is (utils/about-there? (final-answer 0) total-elem-count 3))
     (is (utils/about-there? (final-answer 2) (* 2.0 total-elem-count) 5)))))



(defn dropout-gaussian
  [context]
  (let [batch-size 5
        item-count 100
        repeat-count 30
        input (repeat (* batch-size item-count) 1.0)
        output-gradient (repeat (* batch-size item-count) 2.0)
        test-layer-id :test
        dropout-network (bind-test-network context
                                           [(layers/input item-count)
                                            (layers/multiplicative-dropout 0.5)]
                                           batch-size
                                           {:data item-count
                                            :labels item-count}
                                           test-layer-id)
        answer-seq
        (doall
         (for [iter (range repeat-count)]
           (let [{:keys [output input-gradient]}
                  (forward-backward-bound-network context dropout-network
                                                 input output-gradient
                                                 test-layer-id)]
             [(m/esum (m/eseq output)) (m/esum (m/eseq input-gradient))])))
        final-aggregate  (reduce m/add answer-seq)
        final-answer (m/div final-aggregate repeat-count)
        total-elem-count (double (* item-count batch-size))]
    (is (utils/about-there? (final-answer 0) total-elem-count 10))
    (is (utils/about-there? (final-answer 1) (* 2.0 total-elem-count) 20))))


(defn batch-normalization
  [context]
  (let [batch-size 20
        input-size 20
        input-data-vector-fn (fn []
                               (m/transpose
                                (repeatedly input-size
                                            #(-> (repeatedly batch-size cu/rand-gaussian)
                                                 double-array
                                                 (cu/ensure-gaussian! 5 20)))))
        network (bind-test-network context [(layers/input input-size)
                                            (layers/batch-normalization 0.8)]
                                   batch-size
                                   {:data input-size
                                    :labels input-size}
                                   :test)

        input (input-data-vector-fn)
        output-gradient (repeat (* batch-size
                                   input-size) 1.0)
        {:keys [output]}
        (forward-backward-bound-network context network input output-gradient :test)

        output-batches (mapv vec (partition input-size (m/eseq output)))
        output-stats (mapv cu/calc-mean-variance (m/transpose output-batches))
        input-stats (mapv cu/calc-mean-variance (m/transpose input))]
    (doseq [output-idx (range (count output-stats))]
      (let [{:keys [mean variance]} (output-stats output-idx)]
        (is (utils/about-there? mean 0.0)
            (format "Output mean incorrect at index %s" output-idx))
        (is (utils/about-there? variance 1.0 1e-3)
            (format "Output variance incorrect at index %s" output-idx))))
    (dotimes [iter 5]
      (let [input (input-data-vector-fn)
            {:keys [output]}
            (forward-backward-bound-network context network input output-gradient :test)
            output-batches (mapv vec (partition input-size (m/eseq output)))
            output-stats (mapv cu/calc-mean-variance (m/transpose output-batches))]
       (doseq [output-idx (range (count output-stats))]
         (let [{:keys [mean variance]} (output-stats output-idx)]
           (is (utils/about-there? mean 0.0)
               (format "Output mean incorrect at index %s" output-idx))
           (is (utils/about-there? variance 1.0 1e-3)
               (format "Output variance incorrect at index %s" output-idx))))))


    (let [{:keys [parameters]}
          (forward-backward-bound-network context network input output-gradient :test)
          running-means (get-in parameters [:means :buffer :buffer])
          running-inv-vars (get-in parameters [:variances :buffer :buffer])]
      (is (utils/about-there? 5.0 (/ (m/esum running-means)
                                     input-size)))
      ;;The running variances uses a population calculation for variances
      ;;instead of a specific calculation for variance meaning
      ;;you divide by n-1 instead of n.
      (is (utils/about-there? 21.05 (/ (m/esum running-inv-vars)
                                      input-size)
                              1e-2)))))


(defn- do-lrn-forward
  [context num-input-channels lrn-n]
  (let [batch-size 2
        input-dim 2
        input-num-pixels (* input-dim input-dim)
        n-input (* num-input-channels input-num-pixels)
        input (flatten (repeat batch-size (range n-input)))
        output-gradient (repeat (* batch-size n-input) 1.0)]
    (-> (forward-backward-test context [(layers/input input-dim input-dim num-input-channels)
                                        (layers/local-response-normalization
                                         :n lrn-n :k 1 :alpha 1 :beta 1)]
                               batch-size input output-gradient)
        (assoc :input-data input)
        (update :output m/as-vector))))


(defn lrn-forward
  [context]
  (let [lrn-data (do-lrn-forward context 3 1)]
    (is (m/equals (mapv #(/ (double %) (+ 1 (* % %))) (:input-data lrn-data))
                  (:output lrn-data)
                  1e-4)))
  (let [lrn-data (do-lrn-forward context 3 2)]
    (is (m/equals (mapv double
                        (flatten
                         (repeat 2
                                 [0.0 0.07142857142857142 0.09523809523809523 0.1
                                  0.0975609756097561 0.09259259259259259 0.08695652173913043
                                  0.08139534883720931 0.24242424242424243 0.21686746987951808
                                  0.19607843137254902 0.17886178861788618])))
                  (:output lrn-data)
                  1e-4)))
  (let [lrn-data (do-lrn-forward context 3 3)]
    (is (m/equals (mapv double
                        (flatten
                         (repeat 2
                                 [0.0 0.10344827586206898 0.13953488372093023
                                  0.14754098360655737 0.14457831325301207 0.13636363636363638
                                  0.1258741258741259 0.11538461538461539 0.28915662650602414
                                  0.24770642201834867 0.21582733812949642
                                  0.19075144508670522 ])))
                  (:output lrn-data)
                  1e-4))))
