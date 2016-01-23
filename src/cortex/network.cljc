(ns cortex.network
  "Namespace for constructing and managing neural networks"
  (:require [clojure.core.matrix :as m]
            [cortex.protocols :as cp]
            [clojure.core.matrix.linear :as linear]
            [cortex.optimise :as opt]
            [cortex.core :as core]
            [cortex.util :as util]))

#?(:clj (do
          (set! *warn-on-reflection* true)
          (set! *unchecked-math* true)))

(defn run
  [network test-data]
  (mapv (fn [input]
          (let [network (core/forward network input)]
            (m/clone (core/output network))))
        test-data))

(defn evaluate
  "Evaluate the network assuming we want discreet integers in the result vector"
  [network test-data test-labels]
  (let [double-results (run network test-data)
        integer-results (map (fn [result-row]
                            (m/emap #(Math/round (double %)) result-row))
                             double-results)
        correct (count (filter #(m/equals (first %) (second %)) (map vector integer-results test-labels)))]
    (double (/ correct (count test-data)))))


(defn evaluate-mse
  "evaluate the network using aggregate mse error.  Returns average error over dataset"
  [network test-data test-labels]
  (let [loss-fn (opt/mse-loss)
        results (run network test-data)
        total-error (reduce (fn [sum [result target]]
                              (+ sum (cp/loss loss-fn result target)))
                            0
                            (map vector results test-labels))]
    (/ total-error (count test-data))))

(defn max-index
  [coll]
  (second (reduce (fn [[max-val max-idx] idx]
                    (if (or (nil? max-val)
                            (> (coll idx) max-val))
                      [(coll idx) idx]
                      [max-val max-idx]))
                  [nil nil]
                  (range (count coll)))))

(defn evaluate-softmax
  "evaluate the network assuming single classification and last layer is softmax"
  [network test-data test-labels]
  (let [zeros (apply vector (repeat (first (m/shape (first test-labels))) 0))
        results (mapv #(m/array (assoc zeros (max-index (into [] (seq  %))) 1.0))
                      (run network test-data))
        correct (count (filter #(m/equals (first %) (second %)) (map vector results test-labels)))]
    (double (/ correct (count test-labels)))))


(defn train-step
  [input answer network loss-fn]
  (let [network (core/forward network input)
        temp-answer (core/output network)
        loss (cp/loss loss-fn temp-answer answer)
        loss-gradient (cp/loss-gradient loss-fn temp-answer answer)]
    (core/backward (assoc network :loss loss) input loss-gradient)))


(defn train-batch
  [input-seq label-seq network optimizer loss-fn]
  (let [network (reduce (fn [network [input answer]]
                          (train-step input answer network loss-fn))
                        network
                        (map vector input-seq label-seq))]
    (m/div! (core/gradient network) (double (count input-seq)))
    (core/optimise optimizer network)))

(defn train
  [network optimizer loss-fn training-data training-labels batch-size n-epochs & [test-data test-labels]]
  (let [epoch-batches (repeatedly n-epochs
                                  #(into [] (partition batch-size (shuffle (range (count training-data))))))
        epoch-count (atom 0)
        [optimizer network] (reduce (fn [opt-network batch-index-seq]
                                      (let [_ (swap! epoch-count inc)
                                        ;(println "Running epoch:" @epoch-count)
                                            [optimizer network]
                                            (reduce (fn [[optimizer network] batch-indexes]
                                                      (let [input-seq (mapv training-data batch-indexes)
                                                            answer-seq (mapv training-labels batch-indexes)
                                                            [optimizer network] (train-batch input-seq
                                                                                             answer-seq
                                                                                             network optimizer loss-fn)]
                                                        [optimizer network]))
                                                    opt-network
                                                    batch-index-seq)]
                                        (when test-data
                                          (println "epoch mse-loss:" (evaluate-mse network test-data test-labels)))
                                        [optimizer network]))
                                    [optimizer network]
                                    epoch-batches)]
    network))
