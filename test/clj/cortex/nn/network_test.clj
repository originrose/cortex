(ns cortex.nn.network-test
  (:require
    [clojure.test :refer :all]
    [clojure.core.matrix :as m]
    [cortex.graph :as graph]
    [cortex.nn.layers :as layers]
    [cortex.nn.network :as network]))


(deftest specify-weights-bias
  (let [weight-data [[1 2][3 4]]
        bias-data [0 10]
        built-network (network/linear-network [(layers/input 2)
                                               (layers/linear
                                                2
                                                :weights weight-data
                                                :bias bias-data)])]
    (is (= (vec (m/eseq weight-data))
           (vec (m/eseq (get-in built-network [:compute-graph :buffers
                                               (get-in built-network
                                                       [:compute-graph :nodes
                                                        :linear-1 :weights :buffer-id])
                                               :buffer])))))
    (is (= (vec (m/eseq bias-data))
           (vec (m/eseq (get-in built-network [:compute-graph :buffers
                                               (get-in built-network
                                                       [:compute-graph :nodes
                                                        :linear-1 :bias :buffer-id])
                                               :buffer])))))))


(deftest generate-weights-bias
  (let [bias-data [0 0]
        built-network (network/linear-network [(layers/input 2)
                                               (layers/linear 2)
                                               (layers/relu)])]
    (is (not (nil? (m/eseq (get-in built-network
                                   [:compute-graph :buffers
                                    (get-in built-network
                                            [:compute-graph :nodes
                                             :linear-1 :weights :buffer-id])
                                    :buffer])))))
    (is (m/equals (vec (m/eseq bias-data))
                  (vec (m/eseq (get-in built-network
                                       [:compute-graph :buffers
                                        (get-in built-network
                                                [:compute-graph :nodes
                                                 :linear-1 :bias :buffer-id])
                                        :buffer])))))))



(deftest prelu-initialization
  (let [built-network (network/linear-network [(layers/input 25 25 10)
                                              (layers/prelu)])]
    (is (= (vec (repeat 10 0.25))
           (vec (m/eseq (get-in built-network [:compute-graph :buffers
                                               (get-in built-network
                                                       [:compute-graph :nodes
                                                        :prelu-1 :neg-scale :buffer-id])
                                               :buffer])))))))


(deftest specify-weight-initialization
  (doseq [weight-initialization-type [:relu :xavier :bengio-glorot :orthogonal]]
    (let [built-network (network/linear-network [(layers/input 20 20 3 :id :data)
                                                 (layers/convolutional 2 0 1 2 :weights {:initialization {:type weight-initialization-type}})])]
      (is (= weight-initialization-type
             (get-in built-network [:compute-graph :nodes :convolutional-1 :weights :initialization :type]))))))


(deftest build-concatenate
  (let [network (network/linear-network [(layers/input 25 25 10 :id :right)
                                        (layers/input 500 1 1 :parents [] :id :left)
                                        (layers/concatenate :parents [:left :right] :id :concat)
                                        (layers/linear 10)])
        graph (network/network->graph network)
        concat-node (graph/get-node graph :concat)
        clean-output-dims (fn [node-id]
                           (-> (graph/get-node graph node-id)
                               graph/node->output-dimensions
                               first
                               graph/clear-dimension-identifiers))]
    (is (= (+ (* 25 25 10) 500)
           (graph/node->output-size concat-node)))
    (is (= (set [(assoc (clean-output-dims :right)
                        :id :right)
                 (assoc (clean-output-dims :left)
                        :id :left)])
           (set (graph/node->input-dimensions concat-node))))))


(deftest composite-layer-id
  (let [network (network/linear-network [(layers/input 2 1 1 :id :in)
                                         (layers/linear->softmax 2 :id :out)])
        out-node (network/network->node network :out)]
    (is (= :softmax (:type out-node))))

  (let [network (network/linear-network [(layers/input 2 1 1 :id :in)
                                         (layers/linear->relu 2 :id :out)])
        out-node (network/network->node network :out)]
    (is (= :relu (:type out-node))))

  (let [network (network/linear-network [(layers/input 2 1 1 :id :in)
                                         (layers/linear->tanh 2 :id :out)])
        out-node (network/network->node network :out)]
    (is (= :tanh (:type out-node))))

  (let [network (network/linear-network [(layers/input 2 1 1 :id :in)
                                         (layers/linear->logistic 2 :id :out)])
        out-node (network/network->node network :out)]
    (is (= :logistic (:type out-node)))))
