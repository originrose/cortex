(ns cortex.verify.nn.gradient
  (:require [cortex.nn.execute :as execute]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
            [cortex.nn.traverse :as traverse]
            [cortex.nn.protocols :as cp]
            [cortex.verify.utils :as utils]
            [cortex.verify.nn.layers :as verify-layers]
            [cortex.verify.nn.train :as verify-train]
            [clojure.test :refer :all]
            [cortex.optimise :as opt]
            [cortex.loss :as loss]
            [clojure.core.matrix :as m]
            [cortex.util :as cu]))


(defn close-enough?
  [lhs-vec rhs-vec]
  (empty? (remove #(utils/sig-fig-equal? (first %) (second %))
                  (map vector lhs-vec rhs-vec))))


(defn check-gradients
  [gradient-vector]
  (doseq [{:keys [gradient numeric-gradient buffer-id]} gradient-vector]
    (let [mae (utils/mean-absolute-error (m/as-vector gradient) (m/as-vector numeric-gradient))]
      (is (< mae 1e-6)
          ;;Only build expensive message if we have to.
          (format "Gradient %s failed gradient check (MAE: %s) :\n%s." (name buffer-id) mae
                  (with-out-str
                    (clojure.pprint/pprint
                     (mapv vector
                           (m/eseq gradient)
                           (m/eseq numeric-gradient)))))))))


(defn generate-gradients
  [context network input output loss-fn epsilon batch-size]
  (let [test-id :test
        network (-> network
                    flatten
                    vec)
        num-items (count network)
        input-bindings [(traverse/->input-binding :input :data)]
        output-bindings [(traverse/->output-binding test-id :stream :labels :loss loss-fn)]]
    (as-> (-> network
              (assoc-in [0 :id] :input)
              (assoc-in [(- num-items 1) :id] test-id)) network
      (network/build-network network)
      (traverse/bind-input-bindings network input-bindings)
      (traverse/bind-output-bindings network output-bindings)
      (assoc network :batch-size batch-size)
      (traverse/network->training-traversal network)
      (cp/bind-to-network context network {:numeric-gradients? true})
      (cp/generate-numeric-gradients context network
                                     {:data input
                                      :labels output}
                                     epsilon)
      (verify-layers/unpack-bound-network context network test-id))))


(defn get-gradients
  [& args]
  (let [{:keys [input-gradient numeric-input-gradient parameters] :as gen-result} (apply generate-gradients args)]
    (concat [{:buffer-id :input
              :gradient input-gradient
              :numeric-gradient numeric-input-gradient}]
            (->> parameters
                 (map second)
                 (remove #(get-in % [:description :non-trainable?]))
                 (map (fn [{:keys [description parameter buffer] :as entry}]
                        (merge parameter buffer)))))))


(defn corn-gradient
  [context]
  (let [batch-size 2]
    (-> (get-gradients context
                       [(layers/input 2)
                        (layers/linear 1)]
                       (take batch-size verify-train/CORN-DATA)
                       (take batch-size verify-train/CORN-LABELS)
                       (loss/mse-loss) 1e-4 batch-size)
        check-gradients)))


(defn batch-normalization-gradient
  [context]
  (let [batch-size 10
        input-size 20
        input (repeatedly batch-size
                          #(-> (repeatedly input-size cu/rand-gaussian)
                               double-array
                               (cu/ensure-gaussian! 5 20)))
        output input]
    (-> (get-gradients context
                       [(layers/input input-size)
                        (layers/batch-normalization 1.0)]
                       input output (loss/mse-loss)
                       1e-4 batch-size)
        check-gradients)))


(defn lrn-gradient
  [context]
  (let [batch-size 2
        input-dim 2
        input-num-pixels (* input-dim input-dim)
        num-input-channels 3
        lrn-n 3
        n-input (* num-input-channels input-num-pixels)
        input (flatten (repeat batch-size (range n-input)))
        output input]
    (-> (get-gradients context
                       [(layers/input input-dim input-dim num-input-channels)
                        (layers/local-response-normalization :k 1 :n lrn-n :alpha 1.0 :beta 0.75)]
                       input output (loss/mse-loss)
                       1e-4 batch-size)
        check-gradients)))
