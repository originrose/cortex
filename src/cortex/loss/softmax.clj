(ns cortex.loss.softmax
  (:require [clojure.core.matrix :as m]
            [cortex.util :refer [merge-args max-index]]
            [cortex.compute.math :as math]
            [cortex.compute.nn.backend :as backend]
            [cortex.loss.util :as util]
            [cortex.loss.core :as loss]
            [cortex.graph :as graph]
            [cortex.tensor :as tensor]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Compute implementation
(defn- calculate-cross-entropy-gradient
  [backend v target gradient]
  (tensor/with-stream
    (backend/get-stream)
    (let [target (math/->batch-ct target)
          gradient (math/->batch-ct gradient)
          v (math/->batch-ct v)]
      (tensor/binary-op! gradient 1.0 v 1.0 target :-))))


(defrecord SoftmaxLoss [loss-term backend]
  util/PComputeLoss
  (compute-loss-gradient [this buffer-map]
    (calculate-cross-entropy-gradient backend
                                      (get-in buffer-map [:output :buffer])
                                      (get-in buffer-map [:labels :buffer])
                                      (get-in buffer-map [:output :gradient]))))


(defmethod util/create-compute-loss-term :softmax-loss
  [backend network loss-term batch-size]
  (->SoftmaxLoss loss-term backend))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Graph implementation
(defmethod graph/get-node-metadata :softmax-loss
  [loss-term]
  {:arguments {:output {:gradients? true}
               :labels {}}
   :passes [:loss]})


(defmethod graph/generate-stream-definitions :softmax-loss
  [graph loss-term]
  (util/generate-loss-term-stream-definitions graph loss-term))

(defn log-likelihood-softmax-loss
  ^double [softmax-output answer]
  (let [answer-num (m/esum (m/mul softmax-output answer))]
    (- (Math/log answer-num))))


(defmethod loss/loss :softmax-loss
  [loss-term buffer-map]
  (let [output-channels (long (get loss-term :output-channels 1))
        v (get buffer-map :output)
        target (get buffer-map :labels)]
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


(defmethod loss/generate-loss-term :softmax-loss
  [item-key]
  (util/generic-loss-term item-key))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; util
(defn- softmax-result-to-unit-vector
  [result]
  (let [zeros (apply vector (repeat (first (m/shape result)) 0))]
    (assoc zeros (max-index (into [] (seq result))) 1.0)))


(defn- softmax-results-to-unit-vectors
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
