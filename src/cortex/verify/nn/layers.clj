(ns cortex.verify.nn.layers
  "Verify that layers do actually produce their advertised results."
  (:require
    [clojure.test :refer :all]
    [clojure.core.matrix :as m]
    [think.resource.core :as resource]
    [cortex.graph :as graph]
    [cortex.gaussian :as cu]
    [cortex.nn.layers :as layers]
    [cortex.nn.network :as network]
    [cortex.nn.traverse :as traverse]
    [cortex.nn.execute :as execute]
    [cortex.nn.compute-binding :as compute-binding]
    [cortex.verify.utils :as utils]))


(defn bind-test-network
  [description context batch-size & {:keys [bind-opts]
                                     :or {bind-opts {}}}]
  (let [network (network/linear-network description)]
    (-> network
        (compute-binding/bind-context-to-network
         (execute/current-backend)
         batch-size
         (traverse/training-traversal network :keep-non-trainable? true)
         (assoc bind-opts
                :disable-pooling? true)))))


(defn set-id-and-bind-test-network
  [description context batch-size test-layer-id]
  (let [test-layer-id :test]
    (-> description
      flatten
      vec
      (assoc-in [1 :id] test-layer-id)
      (assoc-in [0 :id] :data)
      (bind-test-network context batch-size))))


(defn unpack-network
  [network test-layer-id]
  (let [traversal (compute-binding/traversal network)
        test-node (get-in network [:compute-graph :nodes test-layer-id])
        parameter-descriptions (->> (graph/get-node-arguments test-node)
                                    (filter #(= :parameter (get % :type))))
        parameters (->> (map (fn [{:keys [key] :as description}]
                               (let [parameter (get test-node key)
                                     buffer (get-in network
                                                    [:compute-graph :buffers
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


(defn unpack-bound-network
  [context network test-layer-id]
  (unpack-network (-> (compute-binding/save-to-network context network {:save-gradients? true})
                      :network)
                  test-layer-id))


(defn- vec->id-val-pairs
  [data-vec id-stem]
  (if (= (count data-vec) 1)
    [[id-stem (first data-vec)]]
    (map-indexed (fn [idx data-item]
                   ;;I know that id generation throughout cortex starts at 1, not 0.
                   [(keyword (format "%s-%s" (name id-stem) (inc idx)))
                    data-item])
                 data-vec)))


(defn vec->stream-map
  "Generate stream id's if the input is not already a map."
  [input-vec stem]
  (if (map? input-vec)
    input-vec
    (->> (vec->id-val-pairs input-vec stem)
         (into {}))))


(defn forward-backward
  "Given a bound network traverse forward and backward using these inputs and
  these output-gradients.  This is used for testing that specific input/output-gradient
  pairs give specific results for layers."
  [bound-network context stream->input-map node-id->output-gradient-map]
  ;;We only provide data for the input streams, we don't provide it for the default loss because
  ;;that loss is not used.
  (let [network-streams (set (->> (network/graph-streams bound-network :inference)
                                  (map first)))
        incoming-streams (set (keys stream->input-map))]
    (when-not (every? incoming-streams
                      network-streams)
     (throw (ex-info "Not all input streams have data"
                     {:bound-streams (clojure.set/intersection network-streams incoming-streams)
                      :unbound-streams (clojure.set/difference network-streams incoming-streams)
                      :incoming-streams incoming-streams}))))
  (let [provided-output-nodes (set (keys node-id->output-gradient-map))
        network-output-nodes (set (network/output-node-ids bound-network :training))]
    (when-not (every? provided-output-nodes network-output-nodes)
      (throw (ex-info "Not all output nodes have provided gradients"
                      {:provided-nodes provided-output-nodes
                       :missing-nodes (clojure.set/difference network-output-nodes provided-output-nodes)}))))
  (let [bound-network (compute-binding/traverse context bound-network stream->input-map :forward)]
    (compute-binding/traverse context bound-network node-id->output-gradient-map :backward)))


(defn forward-backward-bound-network
  [network context input-vec output-gradient-vec test-layer-id]
  (let [input-stream (vec->stream-map input-vec :data)
        output-gradient-stream (vec->stream-map output-gradient-vec test-layer-id)
        network (forward-backward network context input-stream output-gradient-stream)]
    (unpack-bound-network context network test-layer-id)))


(defn forward-backward-test-multiple
  "General test for when there may be a network with multiple inputs or outputs."
  [network context batch-size input-vec output-grad-vec]
  (execute/with-compute-context context
    (-> (bind-test-network network context batch-size)
        (forward-backward-bound-network context input-vec output-grad-vec :test))))


(defn forward-backward-test
  "Given a 2 node network (input,layer) run a test that goes forward and backward
  for that network."
  [description context batch-size input output-gradient]
  (-> (flatten description)
      vec
      (assoc-in [1 :id] :test)
      (assoc-in [0 :id] :data)
      (forward-backward-test-multiple context batch-size
                                      [input] [output-gradient])))


(defn relu-activation
  [context]
  (let [item-count 10
        ;;sums to zero
        {:keys [output input-gradient] :as unpack}
        (forward-backward-test [(layers/input item-count)
                                (layers/relu)]
                               context
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
        (forward-backward-test [(layers/input item-count)
                                (layers/relu)]
                               context
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
        (forward-backward-test [(layers/input item-count)
                                (layers/linear num-output
                                               :weights {:buffer [[1 2] [3 4]]}
                                               :bias {:buffer [0 10]})]
                               context
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
        (forward-backward-test [(layers/input item-count)
                                (layers/linear num-output
                                               :weights {:buffer [[1 2] [3 4]]}
                                               :bias {:buffer [0 10]})]
                               context
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

(comment
;;; calcs to get the activation answers
  (defn sigmoid [v] (/ 1.0
                       (+ 1.0 (Math/exp (- v)))))
  (defn dsigmoid [v] (* v (- 1 v)))
  (defn swish [v] (* v (sigmoid v)))
  (defn fx [v] (* v (sigmoid v)))
  (defn dswish [v] (+  (fx v) (* (sigmoid v) (- 1 (fx v)))))

  (def SELU_ALPHA 1.6732632423543772848170429916717)
  (def SELU_LAMBDA 1.0507009873554804934193349852946)
  (defn selu [x] (if (pos? x)
                   (* SELU_LAMBDA x)
                   (* SELU_LAMBDA (- (* SELU_ALPHA (Math/exp x)) SELU_ALPHA))))

  (defn dselu [x] (if (pos? x)
                    SELU_LAMBDA
                    (* SELU_LAMBDA SELU_ALPHA (Math/exp x))))

  (def input  [-1 1 -1 1 -1 1 -1 1 -1 1])
  (def forward-logistic (mapv sigmoid input))
  (def backward-logistic (->> forward-logistic
                              (map dsigmoid)
                              (map * input)))
  (def forward-swish (mapv swish input))
  (def backward-swish (->> forward-swish
                           (map dswish)
                           (map * input)))

  (def forward-selu (mapv selu input))
  (def backward-selu (->> forward-selu
                          (map dselu)
                          (mapv * input)))


  (def batch-input [-5 -4 -3 -2 -1 0 1 2 3 4])
  (def forward-logistic-batch (mapv sigmoid batch-input))
  (def backward-logistic-batch (->> forward-logistic-batch
                                    (map dsigmoid)
                                    (mapv * batch-input)))
  (def forward-swish-batch (mapv swish batch-input))
  (def backward-swish-batch (->> forward-swish-batch
                                 (map dswish)
                                 (mapv * batch-input)))

  (def forward-selu-batch (mapv selu batch-input))
  (def backward-selu-batch (->> forward-selu-batch
                                 (map dselu)
                                 (mapv * batch-input)))

)


(def activation-answers
  {:logistic [[0.2689414213699951 0.7310585786300049 0.2689414213699951 0.7310585786300049
               0.2689414213699951 0.7310585786300049 0.2689414213699951 0.7310585786300049
               0.2689414213699951 0.7310585786300049]
              [-0.19661193324148185 0.19661193324148185 -0.19661193324148185 0.19661193324148185
               -0.19661193324148185 0.19661193324148185 -0.19661193324148185 0.19661193324148185
               -0.19661193324148185 0.19661193324148185]]
   :swish [[-0.2689414213699951 0.7310585786300049 -0.2689414213699951 0.7310585786300049
            -0.2689414213699951 0.7310585786300049 -0.2689414213699951 0.7310585786300049
            -0.2689414213699951 0.7310585786300049]
           [-0.36713290505919505 0.835403899885462 -0.36713290505919505 0.835403899885462
            -0.36713290505919505 0.835403899885462 -0.36713290505919505 0.835403899885462
            -0.36713290505919505 0.835403899885462]]
   :tanh [[-0.7615941559557649 0.7615941559557649 -0.7615941559557649 0.7615941559557649
           -0.7615941559557649 0.7615941559557649 -0.7615941559557649 0.7615941559557649
           -0.7615941559557649 0.7615941559557649]
          [-0.41997434161402614 0.41997434161402614 -0.41997434161402614 0.41997434161402614
           -0.41997434161402614 0.41997434161402614 -0.41997434161402614 0.41997434161402614
           -0.41997434161402614 0.41997434161402614]]
   :selu [[-1.1113307378125625 1.0507009873554805 -1.1113307378125625 1.0507009873554805
           -1.1113307378125625 1.0507009873554805 -1.1113307378125625 1.0507009873554805
           -1.1113307378125625 1.0507009873554805]
          [-0.5786268790075374 1.0507009873554805 -0.5786268790075374 1.0507009873554805
           -0.5786268790075374 1.0507009873554805 -0.5786268790075374 1.0507009873554805
           -0.5786268790075374 1.0507009873554805]]})

(defn test-activation
  [context act-type]
  (let [item-count 10
        {:keys [output input-gradient]}
        (forward-backward-test [(layers/input item-count)
                                {:type act-type}]
                               context
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
                     0.14130164970632886 0.0295981114963205 0.005363802732103666])))]
   :swish [(vec
               (flatten
                (repeat activation-batch-size
                        [-0.03346425462142428 -0.07194483984836623 -0.14227761953270035
                         -0.2384058440442351 -0.2689414213699951 0.0 0.7310585786300049
                         1.7615941559557646 2.8577223804673 3.928055160151634])))
              (vec
               (flatten
                (repeat activation-batch-size
                        [-2.416354975472916 -1.856234354263459 -1.2873014189882364
                         -0.7638334409883287 -0.36713290505919505 0.0 0.835403899885462
                         2.1475760812469877 3.277268515997532 4.220215568779826])))]

   :selu [(vec
               (flatten
                (repeat activation-batch-size
                        [-1.74625336066962 -1.7258986281898945 -1.6705687287671118
                         -1.520166468595695 -1.1113307378125625 0.0 1.0507009873554805
                         2.101401974710961 3.1521029620664414 4.202803949421922])))
              (vec
               (flatten
                (repeat activation-batch-size
                        [-1.5332932256617793 -1.2518582388020294 -0.9923066122192254
                         -0.768906439142725 -0.5786268790075374 0.0 1.0507009873554805
                         2.101401974710961 3.1521029620664414 4.202803949421922])))]})

(defn test-activation-batch
  [context act-type]
  (let [item-count 10
        batch-size activation-batch-size
        item-range (flatten (repeat batch-size (range (- (/ item-count 2)) (/ item-count 2))))
        {:keys [output input-gradient]}
        (forward-backward-test [(layers/input item-count)
                                {:type act-type}]
                               context
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
        (forward-backward-test [(layers/input item-count)
                                (layers/softmax)]
                               context
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
        (forward-backward-test [(layers/input item-count)
                                (layers/softmax)]
                               context
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


(defn softmax-image
  [context]
  (let [img-dim 5
        batch-size 1
        n-channels 4
        input-range (range 1 (+ n-channels 1))
        input (->> input-range
                   (mapv #(repeat (* img-dim img-dim) %)))
        output-gradient input
        {:keys [output input-gradient]}
        (forward-backward-test [(layers/input img-dim img-dim n-channels)
                                (layers/softmax)]
                               context batch-size input output-gradient)
        output-range [0.03205860328008499 0.08714431874203257
                      0.23688281808991013 0.6439142598879724]]
    (is (m/equals (->> output-range
                       (mapv #(repeat (* img-dim img-dim) %))
                       flatten
                       vec)
                  output
                  1e-4))
    (is (m/equals (vec (flatten input))
                  input-gradient
                  1e-6))))


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
        (forward-backward-test [(layers/input (* channels n-input-pixels))
                                (layers/softmax :output-channels channels)]
                               context
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


(defn conv-layer
  [input-width input-height num-channels k-dim pad stride n-kernels]
  [(layers/input input-width input-height num-channels)
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
                         (repeat (* channel-count batch-size) [1 1 1 1]))
        {:keys [output input-gradient parameters]}
        (forward-backward-test (conv-layer 3 3 1 2 0 1 channel-count)
                               context
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
           (m/eseq input-gradient))))
  (let [batch-size 10
        channel-count 4
        input (repeat batch-size (range 1 13))
        output-gradient (flatten
                         (repeat batch-size (repeat 24 1)))
        {:keys [output input-gradient parameters]}
        (forward-backward-test (conv-layer 3 4 1 2 0 1 channel-count)
                               context
                               batch-size
                               input
                               output-gradient)]
    (is (= (flatten (repeat batch-size [13.0 17.0 25.0 29.0 37.0 41.0 25.0 33.0 49.0 57.0 73.0 81.0 37.0
                                        49.0 73.0 85.0 109.0 121.0 49.0 65.0 97.0 113.0 145.0 161.0] ))
           (m/eseq output)))
    (is (= (map double (flatten (repeat 4 [270.0 330.0 450.0 510.0])))
           (m/eseq (get-in parameters [:weights :buffer :gradient]))))
    (is (= (map double (repeat 4 (* 6 batch-size)))
           (m/eseq (get-in parameters [:bias :buffer :gradient]))))
    (is (= (flatten (repeat batch-size (map double [10.0 20.0 10.0 20.0 40.0 20.0 20.0 40.0 20.0
                                                    10.0 20.0 10.0])))
           (m/eseq input-gradient))))
  (testing "Test bad strides and dims."
    (is (thrown? Exception (layers/convolutional -1 0 1 1)))
    (is (thrown? Exception (layers/convolutional 3 0 -1 1)))))



(defn pool-layer-basic
  [context]
  (execute/with-compute-context context
    (let [batch-size 10
          num-input 16
          input (flatten (repeat batch-size (map inc (range num-input))))
          output-gradient (flatten (repeat batch-size [1 2 3 4]))
          {:keys [output input-gradient]}
          (forward-backward-test [(layers/input 2 2 4)
                                  (layers/max-pooling 2 0 1)]
                                 context
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
            (forward-backward-test [(layers/input 2 2 4)
                                    (layers/max-pooling 2 0 1)]
                                   context
                                   batch-size
                                   input
                                   output-gradient)]
        (is (= (map double (flatten (repeat batch-size [16 12 8 4])))
               (m/eseq output)))
        (is (= (map double (flatten (repeat batch-size (map #(vector % 0 0 0) (range 1 5)))))
               (m/eseq input-gradient)))))))


(defn pool-layer-avg
  [context]
  (execute/with-compute-context context
    (let [batch-size 10
          num-input 16
          num-channels 4
          input (flatten (repeat batch-size (repeat num-input 1)))
          output-gradient (flatten (repeat (* batch-size num-channels) [1 2 3 4 5 6 7 8 9]))
          {:keys [output input-gradient]}
          (forward-backward-test [(layers/input 2 2 num-channels)
                                  (layers/max-pooling 2 1 1 :pool-op :avg)]
                                 context
                                 batch-size
                                 input
                                 output-gradient)]
      (is (= (map double (flatten (repeat (* batch-size num-channels) [0.25 0.5 0.25 0.5 1.0 0.5 0.25 0.5 0.25])))
             (m/eseq output)))
      (is (= (map double (flatten (repeat (* batch-size num-channels) [3 4 6 7])))
             (m/eseq input-gradient))))))


(defn pool-layer-avg-exc-pad
  [context]
  (execute/with-compute-context context
    (let [batch-size 10
          num-input 16
          num-channels 4
          input (flatten (repeat batch-size (repeat num-input 1)))
          output-gradient (flatten (repeat (* batch-size num-channels) [1 2 3 4 5 6 7 8 9]))
          {:keys [output input-gradient]}
          (forward-backward-test [(layers/input 2 2 num-channels)
                                  (layers/max-pooling 2 1 1 :pool-op :avg-exc-pad)]
                                 context
                                 batch-size
                                 input
                                 output-gradient)]
      (is (= (map double (flatten (repeat (* batch-size num-channels) [1 1 1 1 1 1 1 1 1])))
             (m/eseq output)))
      (is (= (map double (flatten (repeat (* batch-size num-channels) [5.25 8.25 14.25 17.25])))
             (m/eseq input-gradient))))))


(defn count-zeros
  [item-seq]
  (count (filter #(= 0.0 (double %)) item-seq)))


(defn dropout-bernoulli
  [context]
  (execute/with-compute-context context
    (let [batch-size 5
          item-count 20
          repeat-count 30
          input (repeat (* batch-size item-count) 1.0)
          output-gradient (repeat (* batch-size item-count) 2.0)
          test-layer-id :test
          dropout-network (set-id-and-bind-test-network [(layers/input item-count)
                                                         (layers/dropout 0.8)]
                                                        context
                                                        batch-size
                                                        test-layer-id)
          answer-seq
          (doall
           (for [iter (range repeat-count)]
             (let [{:keys [output input-gradient]}
                   (forward-backward-bound-network dropout-network context
                                                   [input] [output-gradient]
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
      (is (m/equals (final-answer 0) total-elem-count 3))
      (is (m/equals (final-answer 2) (* 2.0 total-elem-count) 10)))))



(defn dropout-gaussian
  [context]
  (execute/with-compute-context context
   (let [batch-size 5
         item-count 100
         repeat-count 30
         input (repeat (* batch-size item-count) 1.0)
         output-gradient (repeat (* batch-size item-count) 2.0)
         test-layer-id :test
         dropout-network (set-id-and-bind-test-network [(layers/input item-count)
                                                        (layers/multiplicative-dropout 0.5)]
                                                       context
                                                       batch-size
                                                       test-layer-id)
         answer-seq
         (doall
          (for [iter (range repeat-count)]
            (let [{:keys [output input-gradient]}
                  (forward-backward-bound-network dropout-network
                                                  context
                                                  [input] [output-gradient]
                                                  test-layer-id)]
              [(m/esum (m/eseq output)) (m/esum (m/eseq input-gradient))])))
         final-aggregate  (reduce m/add answer-seq)
         final-answer (m/div final-aggregate repeat-count)
         total-elem-count (double (* item-count batch-size))]
     (is (utils/about-there? (final-answer 0) total-elem-count 10))
     (is (utils/about-there? (final-answer 1) (* 2.0 total-elem-count) 20)))))


(defn batch-normalization
  [context]
  (execute/with-compute-context context
   (let [batch-size 20
         input-size 20
         input-data-vector-fn (fn []
                                (m/transpose
                                 (repeatedly input-size
                                             #(-> (repeatedly batch-size cu/rand-gaussian)
                                                  double-array
                                                  (cu/ensure-gaussian! 5 20)))))
         network (set-id-and-bind-test-network [(layers/input input-size)
                                                (layers/batch-normalization
                                                 :ave-factor 0.8)]
                                               context
                                               batch-size
                                               :test)

         input (input-data-vector-fn)
         output-gradient (repeat (* batch-size
                                    input-size) 1.0)
         {:keys [output]}
         (forward-backward-bound-network network context [input] [output-gradient] :test)

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
             (forward-backward-bound-network network context [input] [output-gradient] :test)
             output-batches (mapv vec (partition input-size (m/eseq output)))
             output-stats (mapv cu/calc-mean-variance (m/transpose output-batches))]
         (doseq [output-idx (range (count output-stats))]
           (let [{:keys [mean variance]} (output-stats output-idx)]
             (is (utils/about-there? mean 0.0)
                 (format "Output mean incorrect at index %s" output-idx))
             (is (utils/about-there? variance 1.0 1e-3)
                 (format "Output variance incorrect at index %s" output-idx))))))


     (let [{:keys [parameters]}
           (forward-backward-bound-network network context [input] [output-gradient] :test)
           running-means (get-in parameters [:means :buffer :buffer])
           running-inv-vars (get-in parameters [:variances :buffer :buffer])]
       (is (utils/about-there? 5.0 (/ (m/esum running-means)
                                      input-size)))
       ;;The running variances uses a population calculation for variances
       ;;instead of a specific calculation for variance meaning
       ;;you divide by n-1 instead of n.
       (is (utils/about-there? 21.05 (/ (m/esum running-inv-vars)
                                        input-size)
                               1e-2))))))


(defn- do-lrn-forward
  [context num-input-channels lrn-n]
  (let [batch-size 2
        input-dim 2
        input-num-pixels (* input-dim input-dim)
        n-input (* num-input-channels input-num-pixels)
        input (flatten (repeat batch-size (range n-input)))
        output-gradient (repeat (* batch-size n-input) 1.0)]
    (-> (forward-backward-test [(layers/input input-dim input-dim num-input-channels)
                                (layers/local-response-normalization
                                 :n lrn-n :k 1 :alpha 1 :beta 1)]
                               context
                               batch-size input output-gradient)
        (assoc :input-data input)
        (update :output m/as-vector))))


(defn lrn-forward
  [context]
  (execute/with-compute-context context
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
                    1e-4)))))


(defn prelu
  [context]
  (let [batch-size 10
        channel-count 4
        input-dim 3
        input-size (* input-dim input-dim channel-count)
        input (flatten (repeat batch-size (repeat (quot input-size 2) [-1 1])))
        output-gradient (repeat (* batch-size input-size) 1)
        {:keys [output input-gradient parameters]}
        (forward-backward-test [(layers/input input-dim input-dim channel-count)
                                (layers/prelu)]
                               context
                               batch-size
                               input
                               output-gradient)
        output-answer (->> input
                           (mapv #(if (< % 0)
                                    -0.25
                                    1.0)))]
    (is (= output-answer
           (vec (m/eseq output))))
    (is (= [-10.0 10.0 -10.0 10.0]
           (vec (m/eseq (get-in parameters [:neg-scale :buffer :gradient])))))))


(defn concatenate
  [context]
  (execute/with-compute-context context
    (let [batch-size 4
         item-count 5
         num-inputs 2
         ;;sums to zero
         inputs (->> (partition (* item-count batch-size)
                                (range (* batch-size item-count
                                          num-inputs)))
                     (map #(partition item-count %))
                     (mapv vec)
                     ((fn [input-vec]
                        {:left (first input-vec)
                         :right (second input-vec)})))
         outputs (->> (partition (* item-count batch-size)
                                 (range (* batch-size item-count
                                           num-inputs)))
                      (map #(partition item-count %))
                      (apply interleave)
                      (partition num-inputs)
                      (mapv #(vec (apply concat %))))
         output-gradients outputs
         input-gradients (vec (map inputs [:left :right]))
         network-lr [(layers/input item-count 1 1 :id :right)
                     (layers/input item-count 1 1 :parents [] :id :left)
                     (layers/concatenate :parents [:left :right]
                                         :id :test)]
         {:keys [incoming-buffers outgoing-buffers]}
         (forward-backward-test-multiple network-lr context batch-size
                                         inputs [output-gradients])]
     (is (m/equals input-gradients
                   (mapv :gradient incoming-buffers)))
     (is (m/equals outputs
                   (get-in outgoing-buffers [0 :buffer])))
     (let [network-rl [(layers/input item-count 1 1 :id :right)
                       (layers/input item-count 1 1 :parents [] :id :left)
                       (layers/concatenate :parents [:right :left]
                                           :id :test)]
           swapped-outputs (->> (partition (* item-count batch-size)
                                           (range (* batch-size item-count
                                                     num-inputs)))
                                (map #(partition item-count %))
                                (apply interleave)
                                (partition num-inputs)
                                (mapv #(vec (->> %
                                                 reverse
                                                 (apply concat)))))
           output-gradients swapped-outputs
           {:keys [incoming-buffers outgoing-buffers]}
           (forward-backward-test-multiple network-rl context batch-size
                                           inputs [output-gradients])

           input-gradients (vec (map inputs [:right :left]))]
       (is (m/equals input-gradients
                     (mapv :gradient incoming-buffers)))
       (is (m/equals swapped-outputs
                     (get-in outgoing-buffers [0 :buffer])))))))


(defn split
  [context]
  (execute/with-compute-context context
   (let [input-size 5
         batch-size 5
         output-size 10
         input [(mapv vec (partition input-size (range (* batch-size input-size))))]
         output-gradients {:split-1 (first input)
                           :split-2 (first input)}
         outputs (repeat 2 (first input))
         ;;Split with <2 children acts as a pass-through node.
         {:keys [outgoing-buffers input-gradient]}
         (forward-backward-test-multiple [(layers/input 1 1 input-size :id :data)
                                          (layers/split :id :test)
                                          (layers/split :id :split-1)
                                          (layers/split :parents [:test] :id :split-2)]
                                         context
                                         batch-size
                                         input
                                         output-gradients)]
     (is (m/equals outputs (mapv :buffer outgoing-buffers))))))


;;I can really name something with a -+ suffix?!!
(defn join-+
  [context]
  (let [batch-size 4
        item-count 5
        num-inputs 2
        ;;sums to zero
        input-data (->> (partition (* item-count batch-size)
                                   (range (* batch-size item-count num-inputs)))
                        (map #(partition item-count %))
                        (mapv vec))
        inputs {:left (first input-data)
                :right (second input-data)}
        outputs (->> (apply interleave input-data)
                     (partition num-inputs)
                     (mapv #(apply m/add %)))
        output-gradients (->> (range (* batch-size item-count))
                              (partition item-count)
                              (mapv vec))
        input-gradients (repeat 2 (first input-data))
        {:keys [incoming-buffers outgoing-buffers]}
        (forward-backward-test-multiple [(layers/input item-count 1 1 :id :right)
                                         (layers/input item-count 1 1 :parents [] :id :left)
                                         (layers/join :parents [:left :right]
                                                      :id :test)]
                                        context
                                        batch-size
                                        inputs
                                        [output-gradients])]
    (is (m/equals input-gradients
                  (mapv :gradient incoming-buffers)))
    (is (m/equals outputs
                  (get-in outgoing-buffers [0 :buffer])))))


(defn join-+-2
  [context]
  (let [batch-size 4
        input-counts [3 4 5]
        num-inputs (count input-counts)
        input-sequence (flatten (repeat [-1 -2 1 4]))
        inputs (->> (mapv #(->>
                            (take (* batch-size %) input-sequence)
                            (partition %))
                          input-counts))
        input-map {:left (first inputs)
                   :middle (second inputs)
                   :right (nth inputs 2)}
        output-count (apply max input-counts)
        output-gradients [(->> (repeat (* batch-size output-count) 1)
                               (partition output-count))]
        {:keys [incoming-buffers outgoing-buffers]}
        (forward-backward-test-multiple [(layers/input 3 1 1 :id :left)
                                         (layers/input 4 1 1 :parents [] :id :middle)
                                         (layers/input 5 1 1 :parents [] :id :right)
                                         (layers/join :parents [:left :middle :right]
                                                      :id :test)]
                                        context
                                        batch-size
                                        input-map
                                        output-gradients)]
    ;;You have to pprint the inputs for this output to make any sense
    (is (m/equals [[-3.0,-6.0,3.0,8.0,-1.0],
                   [1.0,-2.0,3.0,3.0,-2.0],
                   [1.0,6.0,-1.0,2.0,1.0],
                   [1.0,-2.0,3.0,5.0,4.0]]
                  (get-in outgoing-buffers [0 :buffer])))))
