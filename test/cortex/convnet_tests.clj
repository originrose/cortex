(ns cortex.convnet-tests
  (:require
    [clojure.test :refer [deftest is are]]
    [cortex.optimise :as opt]
    [clojure.core.matrix :as m]
    [clojure.core.matrix.random :as randm]
    [cortex.util :as util]
    [cortex.network :as net]
    [cortex.core :as core]
    [cortex.layers :as layers]
    [cortex.protocols :as cp]
    [clojure.pprint]))

(m/set-current-implementation :vectorz)

;; Top of train: -0.0037519929582033617,0.08154521439680502 y:1
;; convnet.js:1749 {
;;                  "layers": [
;;                             {
;;                              "out_depth": 2,
;;                              "out_sx": 1,
;;                              "out_sy": 1,
;;                              "layer_type": "input"},
;;                             {
;;                              "out_depth": 2,
;;                              "out_sx": 1,
;;                              "out_sy": 1,
;;                              "layer_type": "fc",
;;                              "num_inputs": 2,
;;                              "l1_decay_mul": 0,
;;                              "l2_decay_mul": 1,
;;                              "filters": [
;;                                          {
;;                                           "sx": 1,
;;                                           "sy": 1,
;;                                           "depth": 2,
;;                                           "w": {
;;                                                 "0": 0.10224438370878293,
;;                                                 "1": 0.06853577379916956},
;;                                           "dw": {
;;                                                  "0": 0,
;;                                                  "1": 0}},
;;                                          {
;;                                           "sx": 1,
;;                                           "sy": 1,
;;                                           "depth": 2,
;;                                           "w": {
;;                                                 "0": 0.17219702650276125,
;;                                                 "1": -0.24396094426599518},
;;                                           "dw": {
;;                                                  "0": 0,
;;                                                  "1": 0}}],
;;                              "biases": {
;;                                         "sx": 1,
;;                                         "sy": 1,
;;                                         "depth": 2,
;;                                         "w": {
;;                                               "0": 0.1,
;;                                               "1": 0.1},
;;                                         "dw": {
;;                                                "0": 0,
;;                                                "1": 0}}},
;;                             {
;;                              "out_depth": 2,
;;                              "out_sx": 1,
;;                              "out_sy": 1,
;;                              "layer_type": "relu"},
;;                             {
;;                              "out_depth": 2,
;;                              "out_sx": 1,
;;                              "out_sy": 1,
;;                              "layer_type": "fc",
;;                              "num_inputs": 2,
;;                              "l1_decay_mul": 0,
;;                              "l2_decay_mul": 1,
;;                              "filters": [
;;                                          {
;;                                           "sx": 1,
;;                                           "sy": 1,
;;                                           "depth": 2,
;;                                           "w": {
;;                                                 "0": -0.05854253476203489,
;;                                                 "1": -0.05741434087406735},
;;                                           "dw": {
;;                                                  "0": 0,
;;                                                  "1": 0}},
;;                                          {
;;                                           "sx": 1,
;;                                           "sy": 1,
;;                                           "depth": 2,
;;                                           "w": {
;;                                                 "0": 0.33518247817877256,
;;                                                 "1": -0.06567875553063794},
;;                                           "dw": {
;;                                                  "0": 0,
;;                                                  "1": 0}}],
;;                              "biases": {
;;                                         "sx": 1,
;;                                         "sy": 1,
;;                                         "depth": 2,
;;                                         "w": {
;;                                               "0": 0,
;;                                               "1": 0},
;;                                         "dw": {
;;                                                "0": 0,
;;                                                "1": 0}}},
;;                             {
;;                              "out_depth": 2,
;;                              "out_sx": 1,
;;                              "out_sy": 1,
;;                              "layer_type": "softmax",
;;                              "num_inputs": 2}]}
;; convnet.js:864 softmax.activation: 0.48981010964882044,0.5101898903511796
;; convnet.js:879 softmax.gradient: 0.48981010964882044,-0.48981010964882044
;; convnet.js:1748 Top of train: -0.004515421423688498,-0.034187375242776985 y:1
;; convnet.js:1749 {
;;                  "layers": [
;;                             {
;;                              "out_depth": 2,
;;                              "out_sx": 1,
;;                              "out_sy": 1,
;;                              "layer_type": "input"},
;;                             {
;;                              "out_depth": 2,
;;                              "out_sx": 1,
;;                              "out_sy": 1,
;;                              "layer_type": "fc",
;;                              "num_inputs": 2,
;;                              "l1_decay_mul": 0,
;;                              "l2_decay_mul": 1,
;;                              "filters": [
;;                                          {
;;                                           "sx": 1,
;;                                           "sy": 1,
;;                                           "depth": 2,
;;                                           "w": {
;;                                                 "0": 0.10224438370878293,
;;                                                 "1": 0.06853577379916956},
;;                                           "dw": {
;;                                                  "0": 0.000723573687069651,
;;                                                  "1": -0.015726034697100124}},
;;                                          {
;;                                           "sx": 1,
;;                                           "sy": 1,
;;                                           "depth": 2,
;;                                           "w": {
;;                                                 "0": 0.17219702650276125,
;;                                                 "1": -0.24396094426599518},
;;                                           "dw": {
;;                                                  "0": -0.000015188044416741875,
;;                                                  "1": 0.0003300945263032887}}],
;;                              "biases": {
;;                                         "sx": 1,
;;                                         "sy": 1,
;;                                         "depth": 2,
;;                                         "w": {
;;                                               "0": 0.1,
;;                                               "1": 0.1},
;;                                         "dw": {
;;                                                "0": -0.19285049176002014,
;;                                                "1": 0.004047993849118164}}},
;;                             {
;;                              "out_depth": 2,
;;                              "out_sx": 1,
;;                              "out_sy": 1,
;;                              "layer_type": "relu"},
;;                             {
;;                              "out_depth": 2,
;;                              "out_sx": 1,
;;                              "out_sy": 1,
;;                              "layer_type": "fc",
;;                              "num_inputs": 2,
;;                              "l1_decay_mul": 0,
;;                              "l2_decay_mul": 1,
;;                              "filters": [
;;                                          {
;;                                           "sx": 1,
;;                                           "sy": 1,
;;                                           "depth": 2,
;;                                           "w": {
;;                                                 "0": -0.05854253476203489,
;;                                                 "1": -0.05741434087406735},
;;                                           "dw": {
;;                                                  "0": 0.051530543196929825,
;;                                                  "1": 0.03892034582692979}},
;;                                          {
;;                                           "sx": 1,
;;                                           "sy": 1,
;;                                           "depth": 2,
;;                                           "w": {
;;                                                 "0": 0.33518247817877256,
;;                                                 "1": -0.06567875553063794},
;;                                           "dw": {
;;                                                  "0": -0.051530543196929825,
;;                                                  "1": -0.03892034582692979}}],
;;                              "biases": {
;;                                         "sx": 1,
;;                                         "sy": 1,
;;                                         "depth": 2,
;;                                         "w": {
;;                                               "0": 0,
;;                                               "1": 0},
;;                                         "dw": {
;;                                                "0": 0.48981010964882044,
;;                                                "1": -0.48981010964882044}}},
;;                             {
;;                              "out_depth": 2,
;;                              "out_sx": 1,
;;                              "out_sy": 1,
;;                              "layer_type": "softmax",
;;                              "num_inputs": 2}]}
;; convnet.js:864 softmax.activation: 0.49065627211341045,0.5093437278865895
;; convnet.js:879 softmax.gradient: 0.49065627211341045,-0.49065627211341045

(def training-data
  [{:x [-0.0037519929582033617 0.08154521439680502] :y 1}
   {:x [-0.004515421423688498 -0.034187375242776985] :y 1 }
   {:x [0.010855556245793329 0.1750381811492602] :y 1}
   {:x [0.01799940757874943 0.09345665135629538] :y 1 }
   {:x [0.038704623641632664 0.15290592085138432] :y 1 }
   {:x [0.09235270735989277 0.21331467353248257] :y 1 }
   {:x [0.13596052324635913 0.2767656532686052] :y 1 }
   {:x [0.17345952701175066 0.26737787723769707] :y 1 }
   {:x [0.26059062778824654 0.4141455028008446] :y 1 }
   {:x [0.264798588569207 0.3454486232763524] :y 1 }])

(defn create-basic-softmax-network
  []
  (let [network (core/stack-module [(layers/linear-layer 2 2)
                                    (layers/relu [2])
                                    (layers/linear-layer 2 2)
                                    (layers/softmax [2])])
        linear-1 (first (:modules network))
        linear-2 (nth (:modules network) 2)
        linear-1 (assoc linear-1 :weights (m/mutable (m/array [[0.10224438370878293 0.06853577379916956]
                                                               [0.17219702650276125 -0.24396094426599518]]))
                        :bias (m/mutable (m/array [0.1 0.1])))
        linear-2 (assoc linear-2 :weights (m/mutable (m/array [[-0.05854253476203489 -0.05741434087406735]
                                                               [0.33518247817877256 -0.06567875553063794]]))
                        :bias (m/mutable (m/array [0.0 0.0])))
        network (assoc network :modules (assoc (:modules network) 0 linear-1 2 linear-2))]
    network
    ))

(deftest basic-softmax
  (let [network (create-basic-softmax-network)
        input (m/array [-0.0037519929582033617 0.08154521439680502])
        network (core/forward network input)
        activation (core/output network)
        gradient (cp/loss-gradient (opt/mse-loss) activation (m/array [0.0 1.0]))
        test-activation (m/array [0.48981010964882044 0.5101898903511796])
        test-gradient (m/array [0.48981010964882044 -0.48981010964882044])
        network (core/backward network input gradient)
        gradients (core/gradient network)
        test-layer1-weight-gradients (m/array [[0.000723573687069651 -0.015726034697100124]
                                               [-0.000015188044416741875 0.0003300945263032887]])
        test-layer1-bias-gradients (m/array [-0.19285049176002014 0.004047993849118164])
        test-layer2-weight-gradients (m/array [[0.051530543196929825 0.03892034582692979]
                                               [-0.051530543196929825 -0.03892034582692979]])
        test-layer2-bias-gradients (m/array [0.48981010964882044 -0.48981010964882044])
        test-gradients (apply m/join (map m/as-vector [test-layer1-weight-gradients test-layer1-bias-gradients test-layer2-weight-gradients test-layer2-bias-gradients]))]

    (is (< (m/distance activation test-activation) 0.0001))
    (is (< (m/distance gradient test-gradient) 0.001))
    (is (< (m/distance gradients test-gradients) 0.001))))


(deftest softmax-train-run
  (let [network (reduce (fn [network input]
                          (let [input-vec (m/array (:x input))
                                network (core/forward network input-vec)
                                activation (core/output network)
                                gradient (cp/loss-gradient (opt/mse-loss) activation (m/array [0.0 1.0]))
                                network (core/backward network input-vec gradient)]
                            network))
                        (create-basic-softmax-network)
                        training-data)
        gradients (core/gradient network)
        test-layer1-weight-gradients (m/array [[-0.18880919033659516 -0.3806211327928244]
                                               [0.0039631656324247136 0.007989359997624613]])
        test-layer1-bias-gradients (m/array [-1.9212542953553482 0.04032774559822303])
        test-layer2-weight-gradients (m/array [[0.6032542126959944 0.3347034488356781]
                                               [-0.6032542126959944 -0.33470344883567815]])
        test-layer2-bias-gradients (m/array [4.879685649141599 -4.879685649141599])
        test-gradients (apply m/join (map m/as-vector [test-layer1-weight-gradients
                                                       test-layer1-bias-gradients
                                                       test-layer2-weight-gradients
                                                       test-layer2-bias-gradients]))]
    (println "gradients:" gradients)
    (println "test-gradients:" test-gradients)
    (is (< (m/distance gradients test-gradients) 0.001))))


(deftest softmax-sgd-update
  (let [network (reduce (fn [network input]
                          (let [input-vec (m/array (:x input))
                                network (core/forward network input-vec)
                                activation (core/output network)
                                gradient (cp/loss-gradient (opt/mse-loss) activation (m/array [0.0 1.0]))
                                network (core/backward network input-vec gradient)]
                            network))
                        (create-basic-softmax-network)
                        training-data)
        gradient (m/div! (core/gradient network) (double (count training-data)))
        [optimizer network] (core/optimise (opt/sgd-optimiser (core/parameter-count network)
                                                              {:learn-rate 0.01 :momentum 0.1})
                                           network)
        linear-1 (first (:modules network))
        test-layer1-weights (m/as-vector (m/array [[0.10243309065473581 0.06891632639618858]
                                                   [0.17219289114010233 -0.24396868966504853]]))
        linear-1-weights (m/as-vector (:weights linear-1))
        test-layer1-biases (m/as-vector (m/array [0.10192125429535535 0.09995967225440178]))
        linear-1-biases (m/as-vector (:bias linear-1))]
    (is (< (m/distance test-layer1-weights linear-1-weights) 0.001))
    (is (< (m/distance test-layer1-biases linear-1-biases) 0.001))
    ))

(deftest core-train-network
  (let [network (create-basic-softmax-network)
        optimizer (opt/sgd-optimiser (core/parameter-count network)
                                     {:learn-rate 0.01 :momentum 0.1})
        training-data (mapv #(m/array (:x %)) training-data)
        training-labels (into [] (repeat 10 (m/array [0 1])))
        loss-fn (opt/mse-loss)
        network (net/train network optimizer loss-fn training-data training-labels (count training-data) 1)
        linear-1 (first (:modules network))
        test-layer1-weights (m/as-vector (m/array [[0.10243309065473581 0.06891632639618858]
                                                   [0.17219289114010233 -0.24396868966504853]]))
        linear-1-weights (m/as-vector (:weights linear-1))
        test-layer1-biases (m/as-vector (m/array [0.10192125429535535 0.09995967225440178]))
        linear-1-biases (m/as-vector (:bias linear-1))]
    (is (< (m/distance test-layer1-weights linear-1-weights) 0.001))
    (is (< (m/distance test-layer1-biases linear-1-biases) 0.001))))


(deftest core-train-network-epoch
  (let [network (create-basic-softmax-network)
        optimizer (opt/sgd-optimiser (core/parameter-count network)
                                     {:learn-rate 0.01 :momentum 0.1})
        training-data (mapv #(m/array (:x %)) training-data)
        training-labels (into [] (repeat 10 (m/array [0 1])))
        loss-fn (opt/mse-loss)
        network (net/train network optimizer loss-fn training-data training-labels (count training-data) 2)
        linear-1 (first (:modules network))
        test-layer1-weights (m/as-vector (m/array [[0.10243309065473581 0.06891632639618858]
                                                   [0.17219289114010233 -0.24396868966504853]]))
        linear-1-weights (m/as-vector (:weights linear-1))
        test-layer1-biases (m/as-vector (m/array [0.10192125429535535 0.09995967225440178]))
        linear-1-biases (m/as-vector (:bias linear-1))]))


(defn compare-networks
  [run-data network]
  (let [linear-1 (nth (:modules network) 0)
        linear-2 (nth (:modules network) 2)
        test-lin-1 (first run-data)
        test-lin-2 (second run-data)
        parameters (core/parameters network)
        test-params (m/array (apply m/join (map m/as-vector [(:weights test-lin-1) (:bias test-lin-1)
                                                              (:weights test-lin-2) (:bias test-lin-2)])))]
    (println "distance: "  (m/dot (m/sub parameters test-params)
                                  (m/sub parameters test-params)))))

(defn create-softmax-network-from-test-data
  [run-data]
  (let [initial-run (first run-data)
        initial-setup (first initial-run)
        source-linear-1 (first initial-setup)
        source-linear-2 (second initial-setup)
        hidden-layer-size (count (:bias source-linear-1))
        network (core/stack-module [(layers/linear-layer 2 hidden-layer-size)
                                    (layers/relu [hidden-layer-size])
                                    (layers/linear-layer hidden-layer-size 2)
                                    (layers/softmax [2])])

        linear-1 (first (:modules network))
        linear-2 (nth (:modules network) 2)
        linear-1 (assoc linear-1 :weights (m/mutable (m/array (:weights source-linear-1)))
                        :bias (m/mutable (m/array (m/array (:bias source-linear-1)))))
        linear-2 (assoc linear-2 :weights (m/mutable (m/array (:weights source-linear-2)))
                        :bias (m/mutable (m/array (:bias source-linear-2))))
        network (assoc network :modules (assoc (:modules network) 0 linear-1 2 linear-2))]
    (compare-networks initial-setup network)
    network))


(defn run-log-test
  [resource-name]
  (let [test-data (read-string (slurp (clojure.java.io/resource resource-name)))
        run-data (into [] (partition 3 (partition 2 (:run test-data))))
        test-input (mapv m/array (partition 2 (:data test-data)))
        test-labels (mapv #(m/array (assoc [0 0] % 1.0)) (:labels test-data))
        network (create-softmax-network-from-test-data run-data)
        optimizer (opt/adadelta-optimiser (core/parameter-count network))
        optimizer (assoc optimizer :decay-rate 0.05 :epsilon 1e-8)
        loss-fn (opt/mse-loss)
        [optimizer network] (reduce (fn [[optimizer network] idx]
                                      (let [input (test-input idx)
                                            network (core/forward network input)
                                            output (core/output network)
                                            gradient (cp/loss-gradient loss-fn output (test-labels idx))
                                            network (core/backward network input gradient)
                                            [optimizer network] (core/optimise optimizer network)
                                            run-result (nth (run-data idx) 2)]
                                        (compare-networks run-result network)
                                        [optimizer network]))
                                    [optimizer network]
                                    (range (count test-input)))]
    (clojure.pprint/pprint (:modules network))
    (clojure.pprint/pprint (net/run network test-input))
    nil))

(deftest adadelta-run
  (run-log-test "adadelta_test.log"))

(deftest spiral-run
  (run-log-test "spiral.log"))
