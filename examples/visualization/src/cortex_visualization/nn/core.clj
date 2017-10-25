(ns cortex-visualization.nn.core
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.macros :refer [c-for]]
            [mikera.image.core :as image]
            [incanter.core :as inc-core]
            [incanter.charts :as inc-charts]
            [cortex.nn.execute :as execute]
            [think.tsne.core :as tsne]))


(defonce a-shift (int 24))
(defonce r-shift (int 16))
(defonce g-shift (int 8))
(defonce b-shift (int 0))

(defn color-int-to-unpacked ^Integer [^Integer px ^Integer shift-amount]
  (int (bit-and 0xFF (bit-shift-right px shift-amount))))

(defn color-unpacked-to-int ^Integer [^Integer bt ^Integer shift-amount]
  (bit-shift-left (bit-and 0xFF bt) shift-amount))

(defn unpack-pixel
  "Returns [R G B A].  Note that java does not have unsigned types
  so some of these may be negative"
  ^Integer [^Integer px]
  [(color-int-to-unpacked px r-shift)
   (color-int-to-unpacked px g-shift)
   (color-int-to-unpacked px b-shift)
   (color-int-to-unpacked px a-shift)])

(defn pack-pixel
  (^Integer [^Integer r ^Integer g ^Integer b ^Integer a]
   (unchecked-int (bit-or
                   (color-unpacked-to-int a a-shift)
                   (color-unpacked-to-int r r-shift)
                   (color-unpacked-to-int g g-shift)
                   (color-unpacked-to-int b b-shift))))
  (^Integer [data]
   (pack-pixel (data 0) (data 1) (data 2) (data 3))))


(defn view-weights
  "Returns a buffered image; currently only works when there is a perfect squre
the row and column counts of the weights are perfect squares."
  [weights]
  (let [filter-count (long (m/row-count weights))
        image-row-width (long (Math/sqrt filter-count))
        image-num-rows (quot filter-count image-row-width)
        filter-dim (Math/sqrt (m/column-count weights))
        ;;separate each filter with a black bar
        filter-bar-offset (+ filter-dim 1)
        num-bars (- image-row-width 1)
        pixel-dim (+ (* image-row-width filter-dim) num-bars)
        num-pixels (* pixel-dim pixel-dim)
        ;;get value range
        filter-min (m/emin weights)
        filter-max (m/emax weights)
        filter-max (Math/max filter-max (Math/abs filter-min))
        filter-range (- filter-max filter-min)
        ^ints image-data (int-array num-pixels)
        retval (image/new-image pixel-dim pixel-dim false)]
    ;;set to black
    (java.util.Arrays/fill image-data (pack-pixel 0 0 0 255))
    (c-for
     [filter-idx 0 (< filter-idx filter-count) (inc filter-idx)]
     (let [filter (m/get-row weights filter-idx)
           image-row (quot filter-idx image-row-width)
           image-column (rem filter-idx image-row-width)
           image-pixel-row-start (* image-row filter-bar-offset)
           image-pixel-column-start (* image-column filter-bar-offset)]
       (c-for
        [filter-row 0 (< filter-row filter-dim) (inc filter-row)]
        (let [pixel-row-offset (+ image-pixel-row-start filter-row)]
          (c-for
           [filter-col 0 (< filter-col filter-dim) (inc filter-col)]
           (let [pixel-col-offset (+ image-pixel-column-start filter-col)
                 pixel-idx (+ (* pixel-row-offset pixel-dim) pixel-col-offset)
                 filter-val (double (m/mget filter (+ (* filter-row filter-dim)
                                                      filter-col)))

                 norm-filter-val (- 1.0 (/ (- filter-val filter-min) filter-range))
                 channel-value (min (unchecked-int (* norm-filter-val 255.0)) 255)
                 pixel (int (pack-pixel channel-value channel-value channel-value 255))]
             (aset image-data pixel-idx pixel)))))))
    (image/set-pixels retval image-data)
    retval))


(defn scatter-plot-data
  "Labels are expected to be an array of integers"
  [data integer-labels title]
  (let [data-transposed (m/transpose data)
        x-vals (m/eseq (m/get-row data-transposed 0))
        y-vals (m/eseq (m/get-row data-transposed 1))]
    (doto (inc-charts/scatter-plot x-vals y-vals :group-by integer-labels :title title)
      inc-core/view)))


(defn index-of-label
  [label-vec]
  (ffirst (filter #(= 1 (long (second %)))
                  (map-indexed vector label-vec))))

(defn plot-network-tsne
  [network data softmax-labels title & {:keys [tsne-iterations]
                                        :or {tsne-iterations 1000}}]
  (let [_ (println "running network:" title)
        samples (execute/run network data)
        labels (mapv index-of-label softmax-labels)
        _ (println (format "tsne-ifying %d samples" (m/row-count data)))
        tsne-data (tsne/tsne samples 2 :iters tsne-iterations)]
    (scatter-plot-data (m/array :vectorz tsne-data) labels title)))
