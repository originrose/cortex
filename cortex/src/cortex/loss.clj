(ns cortex.loss
  "Definitions and implementations of cortex loss functions"
  (:require [clojure.core.matrix :as m]))


(defn arg-list->arg-map
  [args]
  (when-not (= 0 (rem (count args) 2))
    (throw (ex-info "Argument count must be evenly divisble by 2"
                    {:arguments args})))
  (->> (partition 2 args)
       (map vec)
       (into {})))


(defn merge-args
  [desc args]
  (merge desc (arg-list->arg-map args)))


(defn mse-loss
  [& args]
  (merge-args
   {:type :mse-loss}
   args))


(defn softmax-loss
  [& args]
  (merge-args
   {:type :softmax-loss}
   args))


(defmulti loss
  "Implement a specific loss based on the type of the loss function"
  (fn [loss-fn v target]
    (:type loss-fn)))


(defmethod loss :mse-loss
  [loss-fn v target]
  (/ (double (m/magnitude-squared (m/sub v target)))
     (m/ecount v)))


(defn log-likelihood-softmax-loss
  ^double [softmax-output answer]
  (let [answer-num (m/esum (m/mul softmax-output answer))]
    (- (Math/log answer-num))))


(defmethod loss :softmax-loss
  [loss-fn v target]
  (let [output-channels (long (get loss-fn :output-channels 1))]
      (if (= output-channels 1)
        (log-likelihood-softmax-loss v target)
        (let [n-pixels (quot (long (m/ecount v)) output-channels)]
          (loop [pix 0
                 sum 0.0]
            (if (< pix n-pixels)
              (recur (inc pix)
                     (double (+ sum
                                (log-likelihood-softmax-loss
                                 (m/subvector v (* pix output-channels) output-channels)
                                 (m/subvector target (* pix output-channels) output-channels)))))
              (double (/ sum n-pixels))))))))


(defn average-loss
  "V is inferences, target is labels.  Calculate the average loss
across all inferences and labels."
  ^double [loss-fn v-seq target-seq]
  (double
   (/ (->> (map (partial loss loss-fn) v-seq target-seq)
           (reduce +))
      (count v-seq))))


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

(defn evaluate-softmax
  "Provide a percentage correct for softmax.  This is much easier to interpret than
the actual log-loss of the softmax unit."
  [guesses answers]
  (if (or (not (pos? (count guesses)))
          (not (pos? (count answers)))
          (not= (count guesses) (count answers)))
    (throw (Exception. (format "evaluate-softmax: guesses [%d] and answers [%d] count must both be positive and equal."
                               (count guesses)
                               (count answers)))))
  (let [results-answer-seq (mapv vector
                                 (softmax-results-to-unit-vectors guesses)
                                 answers)
        correct (count (filter #(m/equals (first %) (second %)) results-answer-seq))]
    (double (/ correct (count results-answer-seq)))))

(defn center-loss
  "Center loss is a way of specializing an activation for use as a grouping/sorting
mechanism.  It groups activations by class and develops centers for the activations
over the course of training.  Alpha is a number between 1,0 that stands for the exponential
decay factor of the running centers (essentially running means).  The network is penalized
for the distance of the current activations from their respective centers.  The result is that
the activation itself becomes very grouped by class and thus make far better candidates for
LSH or a distance/sorting system.  Note that this is a loss used in the middle of the graph,
not at the edges.
http://ydwen.github.io/papers/WenECCV16.pdf"
  [{:keys [alpha lambda] :as arg-map
    :or {alpha 0.5 lambda 0.01}}]
  (merge {:type :center-loss
          :alpha alpha
          :lambda lambda}))
