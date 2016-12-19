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
   (/ (->> (map (partial loss-fn) v-seq target-seq)
           (reduce +))
      (count v-seq))))
