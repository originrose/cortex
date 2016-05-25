(ns cortex-datasets.math
  (:require [clojure.core.matrix :as m]))

(def epsilon 1e-6)

(defmacro very-near?
  [x y]
  `(< (Math/abs (- ~x ~y)) epsilon))

(defn matrix-col-totals
  [data]
  (let [d-shape (m/shape data)
        col-count (second d-shape)
        totals (reduce (fn [^doubles accum row]
                         (m/add! accum row)
                         accum)
                       (double-array col-count)
                       (m/rows data))]
    totals))


(defn matrix-col-means
  "Calculate the per-column means of the matrix."
  [data]
  (m/div! (matrix-col-totals data) (first (m/shape data))))

(defn matrix-col-variances
  [data means]
  (let [data-shape (m/shape data)
        col-count (second data-shape)
        row-count (first data-shape)
        accumulator (double-array col-count)
        means (or means (double-array col-count))]

    (reduce (fn [accumulator row]
              (let [de-meaned-row (m/sub row means)]
                (m/mul! de-meaned-row de-meaned-row)
                (m/add! accumulator de-meaned-row))
              accumulator)
            accumulator
            data)
    ;;Accumulator now holds the variances
    ;;Assume we are dealing with a sample and not the entire population.
    ;;Although really the row-count of data should be high enough that the
    ;;subtraction is effectively meaningless.
    (when (> row-count 1)
      (m/div! accumulator (- row-count 1)))
    (m/sqrt! accumulator)
    accumulator))


(defn matrix-col-variances
  ([data means]
   (let [data-shape (m/shape data)
         col-count (second data-shape)
         row-count (first data-shape)
         accumulator (double-array col-count)
         means (or means (double-array col-count))]

     (reduce (fn [accumulator row]
               (let [de-meaned-row (m/sub row means)]
                 (m/mul! de-meaned-row de-meaned-row)
                 (m/add! accumulator de-meaned-row))
               accumulator)
             accumulator
             data)
     accumulator))
  ([data] (matrix-col-variances data nil)))


(defn matrix-col-stddevs
  "Calculate per-col stddev.  If the means are nil then a mean of 0 is assumed"
  ([data means]
   (let [data-shape (m/shape data)
         col-count (second data-shape)
         row-count (first data-shape)
         variances (matrix-col-variances data means)]
     ;;Accumulator now holds the variances
     ;;Assume we are dealing with a sample and not the entire population.
     ;;Although really the row-count of data should be high enough that the
     ;;subtraction is effectively meaningless.
     (when (> row-count 1)
       (m/div! variances (- row-count 1)))
     (m/sqrt! variances)
     variances))
  ([data]
   (matrix-col-stddevs data nil)))



(defn normalize-data!
  "In-place normalize the data by column.  This means that all parameters have mean of 0 and
a stddev of either 1 or 0.  This has the effect of making the gradient landscape smoother
and the convergence of the first layer faster due to more efficient weight updates.
This functions does not always work if data is not a vectorz matrix"
  [data]
  (let [col-count (second (m/shape data))
        mat-rows (m/rows data)
        means (matrix-col-means data)
        ;;Set mean to 0
        _ (m/sub! data means)
        ;;Calculate stddev
        stddevs (matrix-col-stddevs data)
        ;;make sure zero entries are set to 1 (no change in data)
        ;;else leave the value.
        _ (m/emap! (fn [^double dval]
                     (if (very-near? dval 0.0)
                       1.0
                       dval))
                   stddevs)
        _ (m/div! data stddevs)]
    {:data data
     :means means
     :stddevs stddevs}))

;;If the chunk size is less than this it probably isn't worth it to break up the data
(def PARALLEL-CHUNK-CUTOFF 5)

(defn parallel-row-ranges
  "Divide the range of rows up into core-count non-overlapping ranges."
  [^long core-count ^long row-count]
  (when (<= core-count 1)
    (throw (Exception. "core-count must be greater than one")))
  (let [chunk-size (quot row-count core-count)
        leftover (rem row-count core-count)]
    (map (fn [^long idx]
           (let [last-index? (= idx (- core-count 1))
                 row-start (* chunk-size idx)
                 row-length (if last-index?
                              (+ chunk-size leftover)
                              chunk-size)]
             [row-start row-length]))
         (range core-count))))

(defn sub-matrix-rows-by-core-count
  [data]
  (let [core-count (.availableProcessors (Runtime/getRuntime))
        data-shape (m/shape data)
        row-count (long (first data-shape))
        col-count (long (second data-shape))
        chunk-size (quot row-count core-count)]
    (if (or (= core-count 1)
            (< chunk-size PARALLEL-CHUNK-CUTOFF))
      [data]
      (let [row-ranges (parallel-row-ranges core-count row-count)]
        (mapv (fn [[row-start row-len]]
               (m/submatrix data row-start row-len 0 col-count))
              row-ranges)))))

(defn parallel-mat-means
  [sub-matrix-seq total-row-count]
  (let [totals (pmap matrix-col-totals sub-matrix-seq)
        final-total (reduce m/add! totals)]
    (m/div! final-total total-row-count)))


(defn parallel-mat-variances
  ([sub-matrix-seq means]
   (let [variances-seq (pmap #(matrix-col-variances % means) sub-matrix-seq)]
     (reduce m/add! variances-seq)))
  ([sub-matrix-seq] (parallel-mat-variances sub-matrix-seq nil)))


(defn parallel-mat-stddevs
  ([sub-matrix-seq row-count means]
   (let [variances (parallel-mat-variances sub-matrix-seq means)]
     (when (> row-count 1)
       (m/div! variances (- row-count 1)))
     (m/sqrt! variances)
     variances))
  ([sub-matrix-seq total-row-count] (parallel-mat-stddevs sub-matrix-seq total-row-count nil)))


(defn parallel-normalize-data!
    "In-place normalize the data by column.  This means that all parameters have mean of 0 and
  a stddev of either 1 or 0.  This has the effect of making the gradient landscape smoother
  and the convergence of the first layer faster due to more efficient weight updates.
  This functions does not always work if data is not a vectorz matrix."
  [data]
  (let [core-count (.availableProcessors (Runtime/getRuntime))
        data-shape (m/shape data)
        row-count (long (first data-shape))
        col-count (long (second data-shape))
        chunk-size (quot row-count core-count)]
    (if (or (= core-count 1)
            (< chunk-size PARALLEL-CHUNK-CUTOFF))
      (normalize-data! data)
      (let [row-ranges (parallel-row-ranges core-count row-count)
            submatrixes (map (fn [[row-start row-len]]
                               (m/submatrix data row-start row-len 0 col-count))
                             row-ranges)
            means (parallel-mat-means submatrixes row-count)
            _ (doall (pmap #(m/sub! % means) submatrixes))
            stddevs (parallel-mat-stddevs submatrixes row-count)
            _ (doall (pmap #(m/div! % stddevs) submatrixes))]
        {:data data
         :means means
         :stddevs stddevs}))))
