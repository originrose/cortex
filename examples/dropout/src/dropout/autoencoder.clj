(ns dropout.autoencoder
  (:require [cortex.description :as desc]
            [cortex.optimise :as opt]
            [cortex.network :as net]
            [cortex-datasets.mnist :as mnist]
            [clojure.core.matrix :as m]
            [mikera.image.core :as image]
            [clojure.core.matrix.macros :refer [c-for]]
            [cortex-gpu.description :as gpu-desc]
            [cortex-gpu.train :as gpu-train]
            [cortex-gpu.resource :as resource]
            [cortex-gpu.cudnn :as cudnn]
            [tsne.core :as tsne]
            [incanter.core :as inc-core]
            [incanter.charts :as inc-charts]))


(def image-count 1000000)
(def training-data (future (vec (take image-count (mnist/training-data)))))
(def training-labels (future (vec (take image-count (mnist/training-labels)))))
(def test-data  (future (vec (take image-count (mnist/test-data)))))
(def test-labels (future (vec (take image-count (mnist/test-labels)))))

(defn build-and-create-network
  [description]
  (gpu-desc/build-and-create-network description))


(defn create-linear
  []
  (let [network-desc [(desc/input 28 28 1)
                      (desc/linear->relu 144)
                      (desc/linear 784)]]
    (build-and-create-network network-desc)))


(defn create-logistic
  []
  (let [network-desc [(desc/input 28 28 1)
                      (desc/linear->logistic 144)
                      (desc/linear 784)]]
    (build-and-create-network network-desc)))



(defn create-dropout-full
  []
  (let [network-desc [(desc/input 28 28 1)
                      (desc/dropout 0.8)
                      (desc/linear->relu 144 :l2-max-constraint 2.0)
                      (desc/dropout 0.5)
                      (desc/linear 784)]]
    (build-and-create-network network-desc)))


(defn create-dropout-logistic-full
  []
  (let [network-desc [(desc/input 28 28 1)
                      (desc/dropout 0.8)
                      (desc/linear->logistic 144 :l2-max-constraint 2.0)
                      (desc/dropout 0.5)
                      (desc/linear 784)]]
    (build-and-create-network network-desc)))


(defn train-and-evaluate
  [network optimiser {:keys [dataset cv-dataset] :as train-config}]
  (let [n-epochs 20
        batch-size 10
        loss-fn (opt/mse-loss)
        full-train-config
        (merge train-config (gpu-train/make-training-config network optimiser loss-fn
                                                            batch-size n-epochs
                                                            dataset cv-dataset))

        train-config (dissoc full-train-config :gpu-cv-dataset)
        train-config (gpu-train/train-uploaded-train-config train-config)
        train-config (assoc train-config :gpu-cv-dataset (:gpu-cv-dataset full-train-config))
        loss (gpu-train/evaluate-training-network train-config)]
    {:network (vec (flatten (desc/network->description network)))
     :score loss}))


(defonce networks (atom nil))


(defn train-network
  [[network-entry opt-entry] train-config]
  (let [[net-name network-fn] network-entry
        [opt-name opt-fn] opt-entry
        network-and-score (train-and-evaluate (network-fn) (opt-fn) train-config)]
    (swap! networks assoc [net-name opt-name]
           network-and-score)))


(defn train-networks
  []
  (let [opt-fns {:adam opt/adam}
        network-fns {:linear create-linear
                     :logistic create-logistic
                     :linear-dropout-full create-dropout-full
                     :logistic-dropout-full create-dropout-logistic-full
                     }
        network-opts (for [opt-fn opt-fns network network-fns]
                       [network opt-fn])]
    (resource/with-resource-context
      (try
       (with-bindings {#'cudnn/*cudnn-datatype* (float 0.0)}
         (let [train-config (gpu-train/upload-datasets
                             {:dataset [@training-data @training-data]
                              :cv-dataset [@test-data @test-data]})]
           ;;Until we start to make sense of cuda streams, there is no benefit
           ;;of a pmap here and I was getting crashes.
           (doall (map #(train-network % train-config) network-opts))))
       (catch Exception e (do (println "caught error: ") (println  e) (throw e)))))
    (map (fn [[key val]]
           [key (:score val)]) @networks)))


(defn check-l2-constraint
  [net-key size]
  (let [network (:network (get @networks [net-key :adam]))
        first-linear-layer (first (filter #(= :linear (get % :type))
                                          network))
        weights (:weights first-linear-layer)
        rows (m/rows weights)
        magnitudes (map m/magnitude rows)
        bad-rows (filter #(> % (+ size 0.00001)) magnitudes)]
    (println (format "found %s filters with magnitudes greater than %f"
                     (count bad-rows), size))))

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
  [net-key]
  (let [network (:network (get @networks [net-key :adam]))
        first-linear-layer (first (filter #(= :linear (get % :type))
                                          network))

        weights (:weights first-linear-layer)
        filter-count (long (m/row-count weights))
        image-row-width (Math/sqrt filter-count)
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
    (image/show retval)
    (image/save retval (str (name net-key) ".jpg"))))

(defn index-of-label
  [label-vec]
  (ffirst (filter #(= 1 (long (second %)))
                  (map-indexed vector label-vec))))

(defn plot-tsne-data
  [tsne-data labels title]
  (let [integer-labels (mapv index-of-label labels)
        tsne-transposed (m/transpose tsne-data)
        x-vals (m/eseq (m/get-row tsne-transposed 0))
        y-vals (m/eseq (m/get-row tsne-transposed 1))]
    (doto (inc-charts/scatter-plot x-vals y-vals :group-by integer-labels :title title)
      inc-core/view)))


(defn run-network
  [net-key & {:keys [sample-count]
              :or {sample-count 2500}}]
  (let [network (:network (get @networks [net-key :adam]))
        first-linear-layer (first (filter #(= :linear (get % :type))
                                          network))
        encoder [(desc/input 28 28 1)
                 first-linear-layer]
        encoder-net (desc/build-and-create-network encoder)
        results (net/run encoder-net (vec (take sample-count @test-data)))]
    results))

(defn plot-network-tsne
  [net-key & {:keys [sample-count iterations]
              :or {sample-count 2500
                   iterations 1000}}]
  (let [_ (println "running network" net-key)
        samples (run-network net-key)
        labels (vec (take sample-count @test-labels))
        _ (println (format "tsne-ifying %d samples" sample-count))
        tsne-data (tsne/tsne samples 2 :iters iterations)]
    (plot-tsne-data tsne-data labels (name net-key))))
