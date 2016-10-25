(ns cortex.metrics
  (:require [clojure.core.matrix :as m]))

(m/set-current-implementation :vectorz)

(defn wrongs
  "Given `y` array of ground truth labels and `y_hat` classifier predictions,
  returns array with 1.0 values where `y` does not equal `y_hat`."
  [y y_hat]
  {:pre [(= (m/shape y) (m/shape y_hat))]}
  (m/emap m/ne y y_hat))

(defn error-rate
  "First argument `y` is the true class, `y_hat` is the predicted value.
  Returns the percentage error rate."
  [y y_hat]
  {:pre [(= (m/shape y) (m/shape y_hat))]}
  (let [wrong (m/esum (wrongs y y_hat))
        len (float (m/ecount y))]
    (/ wrong len)))

(defn accuracy
  "First argument `y` is the true class, `y_hat` is the predicted value.
  Returns the percentage correct."
  [y y_hat]
  {:pre [(= (m/shape y) (m/shape y_hat))]}
  (- 1.0 (error-rate y y_hat)))

(defn false-positives
  "Returns array with 1. values assigned to false positives."
  [y y_hat]
  {:pre [(= (m/shape y) (m/shape y_hat))]}
  (m/emul (wrongs y y_hat) y_hat))

(defn false-negatives
  "Returns array with 1. values assigned to false negatives."
  [y y_hat]
  {:pre [(= (m/shape y) (m/shape y_hat))]}
  (m/emul (wrongs y y_hat) (m/emap (partial m/ne 1.) y)))

(defn true-positives
  "Returns array with 1. values assigned to true positives."
  [y y_hat]
  {:pre [(= (m/shape y) (m/shape y_hat))]}
  (m/emul y y_hat))

(defn recall
  "Returns recall for a binary classifier, a measure of false negative rate"
  [y y_hat]
  {:pre [(= (m/shape y) (m/shape y_hat))]}
  (let [true-count (m/esum y)
        true-pos-count (m/esum (true-positives y y_hat))]
    (/ (float true-pos-count) true-count)))

(defn precision
  "Returns precision for a binary classifier, a measure of false positive rate"
  [y y_hat]
  {:pre [(= (m/shape y) (m/shape y_hat))]}
  (let [true-pos-count (m/esum (true-positives y y_hat))
        false-pos-count (m/esum (false-positives y y_hat))]
    (/ true-pos-count (float (+ true-pos-count false-pos-count)))))

(defn fpr
  "The false negative rate, using the strict ROC definition."
  [y y_hat]
  {:pre [(= (m/shape y) (m/shape y_hat))]}
  (/ (m/esum (false-positives y y_hat))
     (m/esum (m/emap (partial m/ne 1.) y_hat))))

(defn tpr
  "The true positive rate, using the strict ROC definition."
  [y y_hat]
  {:pre [(= (m/shape y) (m/shape y_hat))]}
  (/ (m/esum (true-positives y y_hat))
     (m/esum y_hat)))

(defn fnr
  "The false negative rate, using the strict ROC definition."
  [y y_hat]
  {:pre [(= (m/shape y) (m/shape y_hat))]}
  (- 1 (tpr y y_hat)))

(defn threshold
  "Return a binary mask of all values above threshold."
  [y_est thresh]
  (m/emap m/ge y_est thresh))

(defn unit-space
  "Returns an array with divs+1 values that evenly divide a space from 0.0 to
  1.0, inclusive."
  [divs]
  (m/emap #(/ % divs) (m/array (range (inc divs)))))

(defn- roc-dedupe
  "Dedupes all values for the roc curve in which the same true and false positive
  rates are stored so that we only maintain the (discretized) boundaries for when
  we change rates."
  [triplet-seq]
  (reduce (fn [fp-tp-thresh [fp tp thresh]]
            (let [[fp-prev tp-prev] (last fp-tp-thresh)]
              (if-not (= [fp-prev tp-prev] [fp tp])
                (conj fp-tp-thresh [fp tp thresh])
                fp-tp-thresh)))
          []
          triplet-seq))

(defn roc-curve
  "Compute an ROC curve with `bins` level of discretization for threshold values
  between 0.0 and 1.0 to compute true and false positive rates for.
  
  This is not at all an ideal implementation, just a stand in that is useful
  for certain problems until a real alternative is provided."
  ([y y_est] (roc-curve y y_est 100))
  ([y y_est bins]
   (let [threshold-space (unit-space bins)
         thresholds (remove (fn [th] ; thresholds of all yes or all no create nonsense output
                              (let [pred-count (m/esum (threshold y_est th))]
                                (or (zero? pred-count)
                                    (= pred-count (float (m/ecount y_est))))))
                            threshold-space)
         fprs (map #(fpr y (threshold y_est %)) thresholds)
         tprs (map #(tpr y (threshold y_est %)) thresholds)]
     (roc-dedupe (map vector fprs tprs thresholds)))))

(defn equal-error-point
  "Given y and the continuous, normalized output of a predictor's estimates of
  binary class predictions corresponding to y_hat, select the threshold which
  minimizes the difference between true and false positive rates."
  ([y y_est] (equal-error-point y y_est 100))
  ([y y_est bins]
   (->> (roc-curve y y_est bins)
        (map (fn [[fp tp thresh]]
               [thresh
                (m/abs (- fp (- 1 tp)))]))
        (apply min-key second)
        first)))

(defn eer-accuracy
  "Returns the accuracy where TPR and FPR are balanced, as well as the
  threshold value where this balance is obtained. ROC-EER is the standard
  accuracy measurement in facial recognition."
  ([y y_est] (eer-accuracy y y_est 100))
  ([y y_est bins]
   (let [thresh (equal-error-point y y_est bins)
         y_hat (threshold y_est thresh)]
     {:accuracy (accuracy y y_hat)
      :threshold thresh})))
