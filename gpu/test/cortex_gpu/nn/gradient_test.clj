(ns cortex-gpu.nn.gradient-test
  (:require [clojure.test :refer :all]
            [clojure.core.matrix :as m]
            [cortex.nn.protocols :as cp]
            [cortex.nn.description :as desc]
            [cortex.optimise :as opt]
            [cortex-gpu.nn.cudnn :as cudnn]
            [cortex-gpu.test-framework :refer [def-double-float-test] :as framework]
            [cortex-gpu.nn.train-test :as train-test]
            [cortex-gpu.nn.layers :as layers]
            [cortex-gpu.nn.gradient-check :as grad-check]
            [cortex-gpu.nn.description :as gpu-desc]
            [cortex.gradient-check :as cpu-check]
            [cortex.nn.impl.layers :as cpu-layers]
            )
  )


(use-fixtures :once framework/with-contexts)
(use-fixtures :each framework/with-resource-context)


(defn close-enough?
  [lhs-vec rhs-vec]
  (empty? (remove #(framework/sig-fig-equal? (first %) (second %))
                  (map vector lhs-vec rhs-vec))))


(deftest corn-gradient
  (let [net (layers/linear 2 1)
        loss [(opt/mse-loss)]
        input [(cudnn/array (first train-test/CORN-DATA) 1)]
        output [(cudnn/array (first train-test/CORN-LABELS) 1)]
        net (cp/setup net 1)
        train-config {:network net :loss-fn loss}
        {:keys [calculated-gradients numeric-gradients]} (grad-check/get-gradients train-config input output)]
    (is (close-enough? (nth calculated-gradients 0)
                       (nth numeric-gradients 0)))
    (is (close-enough? (nth calculated-gradients 1)
                       (nth numeric-gradients 1)))))


(deftest corn-gradient-2
  (let [net (gpu-desc/build-and-create-network [(desc/input 2)
                                                (desc/linear 2)
                                                (desc/linear 1)])
        loss [(opt/mse-loss)]
        input [(cudnn/array (first train-test/CORN-DATA) 1)]
        output [(cudnn/array (first train-test/CORN-LABELS) 1)]
        net (cp/setup net 1)
        train-config {:network net :loss-fn loss}
        {:keys [calculated-gradients numeric-gradients]} (grad-check/get-gradients train-config input output)]
    (is (close-enough? (nth calculated-gradients 0)
                       (nth numeric-gradients 0)))
    (is (close-enough? (nth calculated-gradients 1)
                       (nth numeric-gradients 1)))
    (is (close-enough? (nth calculated-gradients 2)
                       (nth numeric-gradients 2)))
    (is (close-enough? (nth calculated-gradients 3)
                       (nth numeric-gradients 3)))))


(deftest softmax-gradient
  (let [net (gpu-desc/build-and-create-network [(desc/input 2)
                                                (desc/linear->logistic 2)
                                                (desc/linear->softmax 2)])
        loss [(opt/softmax-loss)]
        input [(cudnn/array  [-0.0037519929582033617 0.08154521439680502] 1)]
        output [(cudnn/array [0 1] 1)]
        net (cp/setup net 1)
        train-config {:network net :loss-fn loss}
        {:keys [calculated-gradients numeric-gradients]} (grad-check/get-gradients train-config input output)]
    (is (close-enough? (nth calculated-gradients 0)
                       (nth numeric-gradients 0)))
    (is (close-enough? (nth calculated-gradients 1)
                       (nth numeric-gradients 1)))
    (is (close-enough? (nth calculated-gradients 2)
                       (nth numeric-gradients 2)))
    (is (close-enough? (nth calculated-gradients 3)
                       (nth numeric-gradients 3)))))


(deftest dropout-gradient
  (let [net (gpu-desc/build-and-create-network [(desc/input 5)
                                                (desc/linear->logistic 10)
                                                (desc/dropout 0.5)
                                                (desc/linear->softmax 2)])
        loss [(opt/softmax-loss)]
        input [(cudnn/array (vec (repeat 5 1)) 1)]
        output [(cudnn/array [0 1] 1)]
        net (cp/setup net 1)
        {:keys [calculated-gradients numeric-gradients]} (grad-check/get-gradients {:network net :loss-fn loss}
                                                                                   input output)]
    (is (close-enough? (nth calculated-gradients 0)
                       (nth numeric-gradients 0)))
    (is (close-enough? (nth calculated-gradients 1)
                       (nth numeric-gradients 1)))
    (is (close-enough? (nth calculated-gradients 2)
                       (nth numeric-gradients 2)))
    (is (close-enough? (nth calculated-gradients 3)
                       (nth numeric-gradients 3)))))
