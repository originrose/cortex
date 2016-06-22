(ns cortex-gpu.nn.convnet-test
  (:require [clojure.test :refer :all]
            [cortex-gpu.nn.layers :as layers]
            [cortex-gpu.test-framework :as framework]
            [cortex-gpu.nn.cudnn :as cudnn]
            [clojure.core.matrix :as m]
            [cortex.nn.protocols :as cp]))


(use-fixtures :once framework/with-contexts)
(use-fixtures :each framework/with-resource-context)


(defn create-basic-softmax-network
  []
  (let [network
        (layers/layer-list [(layers/->Linear
                             (cudnn/array [[0.10224438370878293 0.06853577379916956]
                                           [0.17219702650276125 -0.24396094426599518]])
                             (cudnn/array [0.1 0.1])
                             nil)
                            (layers/relu 2)
                            (layers/->Linear
                             (cudnn/array [[-0.05854253476203489 -0.05741434087406735]
                                           [0.33518247817877256 -0.06567875553063794]])
                             (cudnn/array [0.0 0.0])
                             nil)
                            (layers/softmax 2)])]
    (cp/setup network 1)))



(deftest basic-softmax
  (let [network (create-basic-softmax-network)
        input (cudnn/array [-0.0037519929582033617 0.08154521439680502])
        network (cp/multi-forward network [input])
        activation (first (cp/multi-output network))
        gradient (cudnn/array [0 0])
        _ (cudnn/loss-gradient 1.0 activation (cudnn/array [0.0 1.0]) gradient)
        test-activation (m/array [0.48981010964882044 0.5101898903511796])
        test-gradient (m/array [0.48981010964882044 -0.48981010964882044])
        network (cp/multi-backward network [input] [gradient])
        gpu-gradients (layers/gradients network)
        gradients (vec (mapcat #(seq (cudnn/to-double-array %)) gpu-gradients))
        test-layer1-weight-gradients (m/array [[0.000723573687069651 -0.015726034697100124]
                                               [-0.000015188044416741875 0.0003300945263032887]])
        test-layer1-bias-gradients (m/array [-0.19285049176002014 0.004047993849118164])
        test-layer2-weight-gradients (m/array [[0.051530543196929825 0.03892034582692979]
                                               [-0.051530543196929825 -0.03892034582692979]])
        test-layer2-bias-gradients (m/array [0.48981010964882044 -0.48981010964882044])
        test-gradients (apply m/join
                              (map m/as-vector [test-layer1-weight-gradients
                                                test-layer1-bias-gradients
                                                test-layer2-weight-gradients
                                                test-layer2-bias-gradients]))]

    (is (< (m/distance (cudnn/to-double-array activation) test-activation) 0.0001))
    (is (< (m/distance (cudnn/to-double-array gradient) test-gradient) 0.001))
    (is (< (m/distance gradients test-gradients) 0.001))))
