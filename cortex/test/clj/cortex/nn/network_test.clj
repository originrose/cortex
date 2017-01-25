(ns cortex.nn.network-test
  (:require [clojure.test :refer :all]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
            [clojure.core.matrix :as m]))



(deftest specify-weights-bias
  (let [weight-data [[1 2][3 4]]
        bias-data [0 10]
        built-network (network/build-network [(layers/input 2)
                                            (layers/linear
                                             2
                                             :weights {:buffer weight-data}
                                             :bias {:buffer bias-data})])]
    (is (= (vec (m/eseq weight-data))
           (vec (m/eseq (get-in built-network [:layer-graph :buffers
                                               (get-in built-network
                                                       [:layer-graph :id->node-map
                                                        :linear-1 :weights :buffer-id])
                                               :buffer])))))
    (is (= (vec (m/eseq bias-data))
           (vec (m/eseq (get-in built-network [:layer-graph :buffers
                                               (get-in built-network
                                                       [:layer-graph :id->node-map
                                                        :linear-1 :bias :buffer-id])
                                               :buffer])))))))



(deftest prelu-initialization
  (let [built-network (network/build-network [(layers/input 25 25 10)
                                              (layers/prelu)])]
    (is (= (vec (repeat 10 0.25))
           (vec (m/eseq (get-in built-network [:layer-graph :buffers
                                               (get-in built-network
                                                       [:layer-graph :id->node-map
                                                        :prelu-1 :neg-scale :buffer-id])
                                               :buffer])))))))
