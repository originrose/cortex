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
            [cortex.optimize :as opt]
            [cortex.loss :as loss]
            [clojure.core.matrix :as m]
            [cortex.gaussian :as gaussian]))


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


(defn add-id-to-desc-list
  [network]
  (let [network (-> network flatten vec)]
   (-> network
       (assoc-in [0 :id] :input)
       (assoc-in [(- (count network) 1) :id] :test))))


(defn generate-gradients
  [context network input output epsilon batch-size]
  (as-> (verify-layers/bind-test-network context network batch-size
                                         (verify-layers/io-vec->stream->size-map
                                          input output batch-size)
                                         :bind-opts {:numeric-gradients? true}) network
    (cp/generate-numeric-gradients context network
                                   (merge (verify-layers/vec->stream-map input :data)
                                          (verify-layers/vec->stream-map output :labels))
                                   epsilon)
    (verify-layers/unpack-bound-network context network :test)))


(defn get-gradients
  [& args]
  (let [{:keys [input-gradient numeric-input-gradient parameters] :as gen-result}
        (apply generate-gradients args)]
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
                       (add-id-to-desc-list
                        [(layers/input 2)
                         (layers/linear 1)])
                       [(take batch-size verify-train/CORN-DATA)]
                       [(take batch-size verify-train/CORN-LABELS)]
                       1e-4 batch-size)
        check-gradients)))


(defn batch-normalization-gradient
  [context]
  (let [batch-size 10
        input-size 20
        input (repeatedly batch-size
                          #(-> (repeatedly input-size gaussian/rand-gaussian)
                               double-array
                               (gaussian/ensure-gaussian! 5 20)))
        output input]
    (-> (get-gradients context
                       (add-id-to-desc-list
                        [(layers/input input-size)
                         (layers/batch-normalization :ave-factor 1.0)])
                       [input] [output]
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
                       (add-id-to-desc-list
                        [(layers/input input-dim input-dim num-input-channels)
                         (layers/local-response-normalization :k 1 :n lrn-n
                                                              :alpha 1.0 :beta 0.75)])
                       [input] [output]
                       1e-4 batch-size)
        check-gradients)))


(defn prelu-gradient
  [context]
  (let [batch-size 10
        input-dim 3
        n-channels 4
        input-num-pixels (* input-dim input-dim)
        n-input (* input-num-pixels n-channels)
        input (flatten (repeat batch-size (repeat (quot n-input 2) [-1 1])))
        output input]
    (-> (get-gradients context
                       (add-id-to-desc-list
                        [(layers/input input-dim input-dim n-channels)
                         (layers/prelu)])
                       [input] [output]
                       1e-4 batch-size)
        check-gradients)))
