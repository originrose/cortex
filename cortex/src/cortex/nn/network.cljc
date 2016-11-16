(ns cortex.nn.network
  "Namespace for constructing and managing neural networks"
  (:require [clojure.core.matrix :as m]
            [cortex.nn.protocols :as cp]
            [clojure.core.matrix.linear :as linear]
            [cortex.optimise :as opt]
            [cortex.nn.core :as core]
            [cortex.util :as util]
            [cortex.nn.serialization :as cs]))

#?(:clj (do
          (set! *warn-on-reflection* true)
          (set! *unchecked-math* true)))

(defn run
  [network test-data]
  (mapv (fn [input]
          (let [network (core/calc network input)]
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
  ^double [network test-data test-labels]
  (let [loss-fn (opt/mse-loss)
        results (run network test-data)
        total-error (reduce (fn [sum [result target]]
                              (+ sum (cp/loss loss-fn result target)))
                            0
                            (map vector results test-labels))]
    (double (/ total-error (count test-data)))))

(defn max-index
  [coll]
  (second (reduce (fn [[max-val max-idx] idx]
                    (if (or (nil? max-val)
                            (> (coll idx) max-val))
                      [(coll idx) idx]
                      [max-val max-idx]))
                  [nil nil]
                  (range (count coll)))))

(defn softmax-result-to-unit-vector
  [result]
  (let [zeros (apply vector (repeat (first (m/shape result)) 0))]
    (assoc zeros (max-index (into [] (seq result))) 1.0)))


(defn softmax-results-to-unit-vectors
  [results]
  (let [zeros (apply vector (repeat (first (m/shape (first results))) 0))]
    (mapv #(assoc zeros (max-index (into [] (seq  %))) 1.0)
          results)))


(defn softmax-results
  ^double [results labels]
  (let [correct (count (filter #(m/equals (first %) (second %))
                               (map vector
                                    (softmax-results-to-unit-vectors results)
                                    labels)))]
    (double (/ correct (count labels)))))


(defn evaluate-softmax
  "evaluate the network assuming single classification and last layer is softmax"
  ^double [network test-data test-labels]
  (softmax-results (run network test-data) test-labels))


(defn train-step
  "Trains a network for a single training example.
   Returns network with updated gradient."
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
    (core/optimise optimizer network (double (count input-seq)))))

(defn train
  [network optimizer loss-fn training-data training-labels batch-size n-epochs & [test-data test-labels]]
  (let [epoch-batches (repeatedly n-epochs
                                  #(into [] (partition batch-size
                                                       (shuffle (range (count training-data))))))
        epoch-count (atom 0)
        [optimizer network]
        (reduce (fn [opt-network batch-index-seq]
                  (let [_ (swap! epoch-count inc)
                        [optimizer network]
                        (reduce (fn [[optimizer network] batch-indexes]
                                  (let [input-seq (mapv training-data batch-indexes)
                                        answer-seq (mapv training-labels batch-indexes)
                                        [optimizer network]
                                        (train-batch input-seq
                                                     answer-seq
                                                     network optimizer loss-fn)]
                                    [optimizer network]))
                                opt-network
                                batch-index-seq)]
                    (when test-data
                      (println "epoch" @epoch-count " mse-loss:" (evaluate-mse network test-data test-labels)))
                    [optimizer network]))
                [optimizer network]
                epoch-batches)]
    network))


(defn train-until-error-stabilizes
  [network optimizer loss-fn training-data training-labels batch-size cv-data cv-labels]
  (loop [network network
         optimizer optimizer
         mean-error-derivative -1.0
         last-error 0.0]
    (if (< mean-error-derivative 0.0)
      (let [epoch-data (vec (partition batch-size batch-size [] (shuffle (range (count training-data)))))
            [optimizer network]
            (reduce (fn [[optimizer network] batch-indexes]
                      (let [input-seq (mapv training-data batch-indexes)
                            answer-seq (mapv training-labels batch-indexes)
                            [optimizer network] (train-batch input-seq
                                                             answer-seq
                                                             network optimizer loss-fn)]
                        [optimizer network]))
                    [optimizer network]
                    epoch-data)
            epoch-error (evaluate-mse network cv-data cv-labels)
            _ (println "epoch error:" epoch-error)
            ;;While this error is smaller than last error, continue
            epoch-derivative (if (= 0.0 last-error)
                               (- epoch-error)
                               (- epoch-error last-error))
            mean-error-derivative (+ (* 0.8 mean-error-derivative) (* 0.2 epoch-derivative))
            last-error epoch-error]
        (recur network optimizer mean-error-derivative last-error))
      network)))

(defn simplified-train-until-error-stabilizes
  [network training-data training-labels
   & {:keys [optimizer loss-fn cv-data cv-labels batch-size noise-fn]
      :or {optimizer (opt/adam)
           loss-fn (opt/mse-loss)
           cv-data training-data
           cv-labels training-labels
           batch-size 10}}]
  (loop [network network
         optimizer optimizer
         mean-error-derivative -1.0
         last-error 0.0]
    (if (< mean-error-derivative 0.0)
      (let [epoch-data (vec (partition batch-size batch-size [] (shuffle (range (count training-data)))))
            [optimizer network]
            (reduce (fn [[optimizer network] batch-indexes]
                      (let [input-seq (mapv training-data batch-indexes)
                            training-input (if noise-fn (mapv noise-fn input-seq) input-seq)
                            answer-seq (mapv training-labels batch-indexes)
                            [optimizer network] (train-batch input-seq
                                                             answer-seq
                                                             network optimizer loss-fn)]
                        [optimizer network]))
                    [optimizer network]
                    epoch-data)
            epoch-error (evaluate-mse network cv-data cv-labels)
            _ (println "epoch error:" epoch-error)
            _ (cs/write-network! network
                                 (clojure.java.io/output-stream "last-network.bin"))
            ;;While this error is smaller than last error, continue
            epoch-derivative (if (= 0.0 last-error)
                               (- epoch-error)
                               (- epoch-error last-error))
            mean-error-derivative (+ (* 0.8 mean-error-derivative) (* 0.2 epoch-derivative))
            last-error epoch-error]
        (recur network optimizer mean-error-derivative last-error))
      network)))
