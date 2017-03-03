(ns cortex.compute.verify.nn.gradient
  (:require [clojure.test :refer :all]
            [cortex.compute.verify.utils :as utils]
            [cortex.compute.nn.gradient-check :as grad-check]
            [cortex.compute.nn.layers :as layers]
            [cortex.compute.nn.backend :as nn-backend]
            [cortex.compute.driver :as drv]
            [cortex.compute.optimize :as opt]
            [cortex.compute.verify.nn.train :as train-test]
            [cortex.compute.math :as math]
            [cortex.util :as cu]))



(defn close-enough?
  [lhs-vec rhs-vec]
  (empty? (remove #(utils/sig-fig-equal? (first %) (second %))
                  (map vector lhs-vec rhs-vec))))

(defn check-gradients
  [{:keys [calculated-gradients numeric-gradients]}
   & {:keys [gradient-count]
      :or {gradient-count (count calculated-gradients)}}]
  (doseq [grad-idx (range gradient-count)]
    (let [calculated-gradients (nth calculated-gradients grad-idx)
          numeric-gradients (nth numeric-gradients grad-idx)
          mae (utils/mean-absolute-error calculated-gradients
                                         numeric-gradients)]
      (is (< mae 1e-6)
          ;;Only build expensive message if we have to.
          (format "Gradient index %s failed gradient check (MAE: %s) :\n%s." grad-idx mae
                  (with-out-str
                    (clojure.pprint/pprint
                     (mapv vector
                           (seq calculated-gradients)
                           (seq numeric-gradients)))))))))


(defn corn-gradient
  [backend]
  (let [batch-size 2
        n-input 2
        n-output 1
        net (layers/linear backend 2 1)
        loss [(opt/setup-loss (opt/mse-loss) backend batch-size
                              (first (cp/multi-output-size net)))]
        input [(nn-backend/array backend (->> (take batch-size train-test/CORN-DATA)
                                           (mapv vec)) batch-size)]
        output [(nn-backend/array backend (->> (take batch-size train-test/CORN-LABELS)
                                            (mapv vec)) batch-size)]
        net (cp/setup net batch-size)
        train-config {:network net :loss-fn loss}]

    (check-gradients (grad-check/get-gradients train-config
                                               input output))))

(defn softmax-gradient
  [backend]
  (let [batch-size 1
        net (layers/layer-list [(layers/linear backend 2 2)
                                (layers/softmax backend 2)])
        loss [(opt/setup-loss (opt/softmax-loss) backend batch-size 2)]
        input [(nn-backend/array  backend [-0.0037519929582033617 0.08154521439680502]
                                  batch-size)]
        output [(nn-backend/array backend [0 1] batch-size)]
        net (cp/setup net 1)
        train-config {:network net :loss-fn loss}]
    ;;Softmax isn't sensitive to the input gradient changing by a small epsilon
    ;;so checking the input gradient doesn't work.
    (check-gradients (grad-check/get-gradients train-config
                                               input output)
                     :gradient-count 2)))


(defn dropout-gaussian-gradient
  [backend]
  (let [batch-size 1
        net (layers/->LayerList [(layers/linear backend 5 10)
                                 (layers/activation backend 10 :sigmoid)
                                 (layers/gaussian-dropout backend 10 0.5)
                                 (layers/linear backend 10 2)
                                 (layers/softmax backend 2)])
        loss [(opt/setup-loss (opt/softmax-loss) backend batch-size
                              (first (cp/multi-output-size net)))]
        input [(nn-backend/array backend (vec (repeat 5 1)) batch-size)]
        output [(nn-backend/array backend [0 1] batch-size)]
        net (cp/setup net 1)]
    (check-gradients (grad-check/get-gradients {:network net :loss-fn loss}
                                               input output)
                     :gradient-count 4)))


(defn bn-gradient
  [backend]
  (let [batch-size 10
        input-size 20
        input-data-vector (repeatedly batch-size
                                      #(-> (repeatedly input-size cu/rand-gaussian)
                                           double-array
                                           (cu/ensure-gaussian! 5 20)))
        layer (cp/setup
               (layers/batch-normalization backend input-size 1.0)
               batch-size)
        input [(nn-backend/array backend input-data-vector batch-size)]
        output [(nn-backend/array backend input-data-vector batch-size)]
        loss [(opt/setup-loss (opt/mse-loss) backend batch-size
                              (first (cp/multi-output-size layer)))]]
    ;;Set the input epsilon to something pretty large because with MSE error each
    ;;individual input value has a small effect on the output and an epsilon of 1e-4
    ;;means we get gradients in the range of 1e-7 which makes the tests fail.
    (check-gradients (grad-check/get-gradients {:network layer :loss-fn loss}
                                               input output))))


(defn lrn-gradient
  [backend]
  (let [batch-size 2
        input-dim 2
        input-num-pixels (* input-dim input-dim)
        num-input-channels 3
        lrn-n 3
        n-input (* num-input-channels input-num-pixels)
        input-data (flatten (repeat batch-size (range n-input)))
        input [(math/with-tensor
                 (nn-backend/array backend input-data batch-size)
                 (math/map->Tensor {:batch-size batch-size
                                    :channel-count num-input-channels
                                    :width input-dim
                                    :height input-dim} ))]
        output [(math/with-tensor
                  (nn-backend/array backend input-data batch-size)
                  (math/map->Tensor {:batch-size batch-size
                                     :channel-count num-input-channels
                                     :width input-dim
                                     :height input-dim} ))]
        layer (cp/setup (layers/local-response-normalization
                         backend
                         input-dim input-dim num-input-channels
                         :k 1 :n lrn-n :alpha 1.0 :beta 0.75)
                        batch-size)
        loss [(opt/setup-loss (opt/mse-loss) backend batch-size n-input)]]
    (check-gradients (grad-check/get-gradients {:network layer :loss-fn loss}
                                               input output))))
