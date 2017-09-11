(ns cortex.metrics
  (:require [clojure.core.matrix :as m]))

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
  (m/eq 1 (m/emap - y_hat y)))

(defn false-negatives
  "Returns array with 1. values assigned to false negatives."
  [y y_hat]
  {:pre [(= (m/shape y) (m/shape y_hat))]}
  (m/eq 1 (m/emap - y y_hat)))

(defn true-positives
  "Returns array with 1. values assigned to true positives."
  [y y_hat]
  {:pre [(= (m/shape y) (m/shape y_hat))]}
  (m/emul y y_hat))

(defn true-negatives
  "Returns array with 1. values assigned to true negatives."
  [y y_hat]
  {:pre [(= (m/shape y) (m/shape y_hat))]}
  (m/eq 0 (m/emap + y y_hat)))

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

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;; METRICS FOR LOCALIZATION WITH CLASSIFICATION ;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn- map-values
  "Apply a function val-fn element-wise the the values of a hashmap hm."
  [val-fn hm]
  (zipmap (keys hm) (map val-fn (vals hm))))

(defn- bb-matches
  "Returns the list of label-prediction pairs that have a 'good' bounding box match with one another,
  where 'good' is detemined by the iou function and the iou-threshold.
  How the list is built:
  1. Calculate a list IOU containing the intersection over union (iou) for each label-prediction pair (l, p).
  2. Add the pair (l,p) with highest iou to the list C of matches.
  3. Remove any pairs from IOU that have either l or p in them, since those boxes can only be used once.
  4. If IOU is non-empty return to step 2, otherwise return C."
  [labels predictions iou-fn iou-threshold]
  (let [IOU (->> (for [l labels p predictions] {:iou (iou-fn l p) :label l :prediction p})
                 (remove #(> iou-threshold (:iou %)))
                 (sort-by :iou >))]
    (loop [M IOU C []]
      (if (empty? M)
        C
        (let [match (first M)]
          (recur (remove #(or (= (:label match) (:label %))
                              (= (:prediction match) (:prediction %))) M)
                 (conj C match)))))))


(defn- per-class-metrics
  "Returns the sensitivity, precision, and F1 scores given labels and predictions, which are assumed to be of the same class c.
   Since finding and classifying objects is not a binary valued test, these are only analogues of the well-known sensitivity, precision and F1 for binary tests.
   The definitions are as follows:
  Location Sensitivity = #(correctly located and classified objects with class c) / #(labels with class c)
  Location Precision = #(correctly located and classified objects with class c) / #(predictions with class c)
  Location F1 = harmonic mean of sensitivity and precision."
  [labels predictions iou-fn iou-threshold]
  (let [bb-matches (bb-matches labels predictions iou-fn iou-threshold)
        label-count (count labels)
        pred-count (count predictions)
        bb-count (count bb-matches)]
    (map-values double
                {:location-sensitivity (if (= 0 label-count)
                                         1
                                         (/ bb-count label-count))
                 :location-precision   (if (= 0 pred-count)
                                         1
                                         (/ bb-count pred-count))
                 :location-F1          (if (= 0 (+ label-count pred-count))
                                         1
                                         (/ (* 2 bb-count) (+ label-count pred-count)))})))

(defn- global-metrics
  "Returns 5 numbers:
  1. Location Sensitivity (also called RECALL) = #(bb-matches) / #(labels)
  2. Location Precision = #(bb-matches) / #(predictions)
  3. Location F1 = harmonic mean of (1) and (2) = 2 * #(bb-matches) / #(labels) + #(predictions)
  4. Classification accuracy = #(bb-matches with correct class) / #(bb-matches)
  5. Global F1 = (Location F1) * (Classification accuracy) = 2 * #(bb-matches with correct class) / #(labels) + #(predictions)"
  [labels predictions label->class-fn iou-fn iou-threshold]
  (let [bb-matches (bb-matches labels predictions iou-fn iou-threshold)
        bb-matches-with-correct-class (filter #(= (label->class-fn (:label %)) (label->class-fn (:prediction %))) bb-matches)
        label-count (count labels)
        pred-count (count predictions)
        bb-count (count bb-matches)
        bb-with-class-count (count bb-matches-with-correct-class)]
    (map-values double
                {:location-sensitivity    (if (= 0 label-count)
                                            1
                                            (/ bb-count label-count))
                 :location-precision      (if (= 0 pred-count)
                                            1
                                            (/ bb-count pred-count))
                 :location-F1             (if (= 0 (+ label-count pred-count))
                                            1
                                            (/ (* 2 bb-count) (+ label-count pred-count)))
                 :classification-accuracy (if (= 0 bb-count)
                                            1
                                            (/ bb-with-class-count bb-count))
                 :global-F1               (if (= 0 (+ label-count pred-count))
                                            1
                                            (/ (* 2 bb-with-class-count) (+ label-count pred-count)))})))

(defn all-metrics
  "Returns global and per-class metrics for a given set of labels and predictions.
  - label->class-fn should take a label or prediction and return the class as a string or keyword.
  - iou-fn should take a label and prediction and return the intersection over union score
  - iou-threshold determines what iou value constitutes a matching bounding box.
  ** NOTE: If labels and predictions are produced from a sequence of images,
     ensure that the bounding boxes are shifted in each image so that there is not an overlap."
  [labels predictions label->class-fn iou-fn iou-threshold]
  (let [labels-by-class      (group-by label->class-fn labels)
        predictions-by-class (group-by label->class-fn predictions)],
    (merge {:global-metrics (global-metrics labels predictions label->class-fn iou-fn iou-threshold)}
           {:per-class-metrics
            (for [class (sort (keys labels-by-class))]
              (merge {:class class}
                     (per-class-metrics
                       (get labels-by-class class)
                       (get predictions-by-class class)
                       iou-fn
                       iou-threshold)))})))

