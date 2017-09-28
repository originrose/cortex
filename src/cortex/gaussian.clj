(ns cortex.gaussian
  "Namespace for working with gaussian distributions."
  (:require [clojure.core.matrix :as m])
  (:import [java.security SecureRandom]))


(def ^:dynamic ^SecureRandom *RAND-GENERATOR* (SecureRandom.))

(defn rand-normal ^double []
  (.nextDouble *RAND-GENERATOR*))

(defn rand-gaussian ^double []
  (.nextGaussian *RAND-GENERATOR*))


(defn calc-mean-variance
  [data]
  (let [num-elems (double (m/ecount data))
        elem-sum (double (m/esum data))]
    (if (= num-elems 0.0)
      {:mean 0
       :variance 0}
      (let [mean (/ elem-sum num-elems)
            variance (/ (double (m/ereduce (fn [^double sum val]
                                             (let [temp-val (- mean (double val))]
                                               (+ sum (* temp-val temp-val))))
                                           0.0
                                           data))
                        num-elems)]
        {:mean mean
         :variance variance}))))


(defn ensure-gaussian!
  [data ^double mean ^double variance]
  (let [actual-stats (calc-mean-variance data)
        actual-variance (double (:variance actual-stats))
        actual-mean (double (:mean actual-stats))
        variance-fix (Math/sqrt (double (if (> actual-variance 0.0)
                                          (/ variance actual-variance)
                                          1.0)))
        adjusted-mean (* actual-mean variance-fix)
        mean-fix (- mean adjusted-mean)]
    (doall (m/emap! (fn [^double data-var]
                      (+ (* variance-fix data-var)
                         mean-fix))
                    data))))
