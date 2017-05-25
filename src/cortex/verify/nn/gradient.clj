(ns cortex.verify.nn.gradient
  (:require [cortex.nn.execute :as execute]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
            [cortex.nn.traverse :as traverse]
            [cortex.verify.utils :as utils]
            [cortex.verify.nn.layers :as verify-layers]
            [cortex.verify.nn.data :as verify-data]
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
  [context description input output epsilon batch-size labels-key]
  (execute/with-compute-context context
    (-> (network/linear-network description)
        (execute/generate-numeric-gradients context batch-size
                                            (merge (verify-layers/vec->stream-map input :data)
                                                   (verify-layers/vec->stream-map output labels-key))
                                            epsilon)
        (verify-layers/unpack-network :test))))


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
                       [(layers/input 1 1 2 :id :data)
                        (layers/linear 1 :id :test)]
                       [(take batch-size verify-data/CORN-DATA)]
                       [(take batch-size verify-data/CORN-LABELS)]
                       1e-4 batch-size :test)
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
                       [(layers/input 1 1 input-size :id :data)
                        (layers/batch-normalization :ave-factor 1.0 :id :test)]
                       [input] [output]
                       1e-4 batch-size :test)
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
                       [(layers/input input-dim input-dim num-input-channels :id :data)
                        (layers/local-response-normalization :k 1 :n lrn-n
                                                             :alpha 1.0 :beta 0.75
                                                             :id :test)]
                       [input] [output]
                       1e-4 batch-size :test)
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
                       [(layers/input input-dim input-dim n-channels :id :data)
                        (layers/prelu :id :test)]
                       [input] [output]
                       1e-4 batch-size :test)
        check-gradients)))


(defn concat-gradient
  [context]
  (let [batch-size 4
        item-count 5
        num-inputs 2
        ;;sums to zero
        inputs (->> (partition (* item-count batch-size)
                               (range (* batch-size item-count
                                         num-inputs)))
                    (map #(partition item-count %))
                    (mapv vec)
                    ((fn [input-val]
                       {:right (first input-val)
                        :left (second input-val)})))

        outputs [(repeat (* item-count num-inputs batch-size) 1)]]
    (-> (get-gradients context
                       [(layers/input item-count 1 1 :id :right)
                        (layers/input item-count 1 1 :parents [] :id :left)
                        (layers/concatenate :parents [:left :right]
                                            :id :test)]
                       inputs outputs
                       1e-4 batch-size :test)
        check-gradients)))

(defn split-gradient
  [context]
  (let [batch-size 4
        input-size 5
        num-outputs 2
        inputs [(mapv vec (partition input-size (range (* batch-size input-size))))]
        outputs (->> (repeat num-outputs (repeat (* input-size batch-size) 1))
                     ((fn [output-vec]
                        {:output-1 (first output-vec)
                         :output-2 (second output-vec)})))]
    (-> (get-gradients context
                       [(layers/input input-size)
                        (layers/split :id :test)
                        (layers/split :id :output-1)
                        (layers/split :parents [:test] :id :output-2)]
                       inputs outputs
                       1e-4 batch-size :test)
        check-gradients)))


(defn join-+-gradient
  [context]
  (let [batch-size 4
        input-counts [3 4 5]
        num-inputs (count input-counts)

        input-sequence (flatten (repeat [-1 -2 1 4]))
        inputs (->> (mapv #(->>
                            (take (* batch-size %) input-sequence)
                            (partition %))
                          input-counts)
                    ((fn [input-vec]
                       {:left (first input-vec)
                        :middle (second input-vec)
                        :third (nth input-vec 2)})))
        output-count (apply max input-counts)
        outputs [(->> (repeat (* batch-size output-count) 1)
                      (partition output-count))]]
    (-> (get-gradients context
                       [(layers/input 3 1 1 :id :left)
                        (layers/input 4 1 1 :parents [] :id :middle)
                        (layers/input 5 1 1 :parents [] :id :right)
                        (layers/join :parents [:left :middle :right]
                                     :id :test)]
                       inputs outputs
                       1e-4 batch-size :test)
        check-gradients)))


(defn join-*-gradient
  [context]
  (let [batch-size 4
        input-counts [3 4 5]
        num-inputs (count input-counts)
        ;;sums to zero
        input-sequence (flatten (repeat [-1 -2 1 4]))
        inputs (->> (mapv #(->>
                            (take (* batch-size %) input-sequence)
                            (partition %))
                          input-counts)
                    ((fn [input-vec]
                       {:left (first input-vec)
                        :middle (second input-vec)
                        :third (nth input-vec 2)})))
        output-count (apply max input-counts)
        outputs [(->> (repeat (* batch-size output-count) 1)
                      (partition output-count))]]
        (-> (get-gradients context
                           [(layers/input 3 1 1 :id :left)
                            (layers/input 4 1 1 :parents [] :id :middle)
                            (layers/input 5 1 1 :parents [] :id :right)
                            (layers/join :parents [:left :middle :right]
                                         :operation :*
                                         :id :test)]
                           inputs outputs
                           1e-4 batch-size :test)
            check-gradients)))

(defn censor-gradient
  [context]
  (let [batch-size 1
        inputs [[4.0 6.0]]
        outputs [[-0.5 Double/NaN]]
        input-dim (count (first inputs))
        output-dim (count (first outputs))
        network [(layers/input input-dim 1 1 :id :data)
                 (layers/linear output-dim
                                :id :test
                                :censor-loss {:labels {:type :stream
                                                       :stream :test}
                                              :gradient-masks {:stream :test}
                                              :gradient-multi-masks {:stream :test}})]
        _ (println "Starting...")
        gradients (get-gradients context
                                 network
                                 [inputs]
                                 [outputs]
                                 1e-4
                                 batch-size
                                 :test)]
    (check-gradients gradients)))
