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
