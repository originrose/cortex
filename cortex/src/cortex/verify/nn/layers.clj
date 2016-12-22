(ns cortex.verify.nn.layers
  "Verify that layers do actually produce their advertised results."
  (:require [cortex.nn.layers :as layers]
            [cortex.nn.build :as build]
            [cortex.nn.traverse :as traverse]
            [cortex.nn.execute :as execute]
            [clojure.test :refer :all]
            [clojure.core.matrix :as m]
            [cortex.verify.utils :as utils]))


(defn forward-backward-test
  "Given a 2 node network (input,layer) run a test that goes forward and backward
for that network."
  [context network batch-size input output-gradient]
  (let [test-layer-id :test
        network (-> network
                    flatten
                    vec
                    (assoc-in [0 :id] :input)
                    (assoc-in [1 :id] test-layer-id))
        input-bindings {:input :data}
        output-bindings {test-layer-id {:stream :labels}}
        input-stream {:data input}
        output-gradient-stream {:test output-gradient}
        network (execute/forward-backward context
                                          (assoc (traverse/network->training-traversal (build/build-network network)
                                                                                       input-bindings
                                                                                       output-bindings)
                                                 :batch-size
                                                 batch-size)
                                          input-stream output-gradient-stream)
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
     :input-gradient (get (first incoming-buffers) :gradient)}))


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
