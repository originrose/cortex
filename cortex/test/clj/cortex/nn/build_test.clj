(ns cortex.nn.build-test
  (:require [clojure.test :refer :all]
            [cortex.nn.layers :as layers]
            [cortex.nn.build :as build]
            [clojure.core.matrix :as m]))



(deftest specify-weights-bias
  (let [weight-data [[1 2][3 4]]
        bias-data [0 10]
        built-network (build/build-network [(layers/input 2)
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
