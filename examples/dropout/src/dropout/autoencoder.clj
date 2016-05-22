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


(def training-data (future (mnist/training-data)))
(def training-labels (future (mnist/training-labels)))
(def test-data  (future (mnist/test-data)))
(def test-labels (future (mnist/test-labels)))
(def autoencoder-size 529)

(defn build-and-create-network
  [description]
  (gpu-desc/build-and-create-network description))


(defn single-target-network
  [network-desc]
  {:network-desc network-desc
   :loss-fn (opt/mse-loss)
   :labels {:training @training-data :test @test-data}})


(defn create-linear
  []
  (let [network-desc [(desc/input 28 28 1)
                      (desc/linear->relu autoencoder-size)
                      (desc/linear 784)]]
    (single-target-network network-desc)))


(defn create-logistic
  []
  (let [network-desc [(desc/input 28 28 1)
                      (desc/linear->logistic autoencoder-size)
                      (desc/linear 784)]]
    (single-target-network network-desc)))



(defn create-dropout-full
  []
  (let [network-desc [(desc/input 28 28 1)
                      (desc/dropout 0.8)
                      (desc/linear->relu autoencoder-size :l2-max-constraint 2.0)
                      (desc/dropout 0.5)
                      (desc/linear 784)]]
    (single-target-network network-desc)))


(defn create-dropout-logistic-full
  []
  (let [network-desc [(desc/input 28 28 1)
                      (desc/dropout 0.8)
                      (desc/linear->logistic autoencoder-size :l2-max-constraint 2.0)
                      (desc/dropout 0.5)
                      (desc/linear 784)]]
    (single-target-network network-desc)))

(defn create-multi-target-dropout
  []
  (let [network-desc [(desc/input 28 28 1)
                      (desc/dropout 0.8)
                      (desc/linear->logistic autoencoder-size :l2-max-constraint 2.0)
                      (desc/dropout 0.5)
                      ;;Note carefully the order of the leaves of the network.  There
                      ;;is currently an implicit dependency here on that order, the order
                      ;;of the loss functions and the order of the training and test
                      ;;labels which is probably an argument to specify all of that
                      ;;in the network description
                      (desc/split [[(desc/linear 784)] [(desc/linear->softmax 10)]])]
        labels {:training [@training-data @training-labels]
                :test [@test-data @test-labels]}
        loss-fn [(opt/mse-loss) (opt/softmax-loss)]]
    {:network-desc network-desc
     :labels labels
     :loss-fn loss-fn}))

(defn create-mnist-multi-target
  []
  (let [network-desc [(desc/input 28 28 1)
                      (desc/dropout 0.8)
                      (desc/convolutional 5 0 1 20)
                      (desc/max-pooling 2 0 2)
                      (desc/convolutional 5 0 1 50)
                      (desc/max-pooling 2 0 2)
                      (desc/linear->logistic autoencoder-size :l2-max-constraint 2.0)
                      (desc/dropout 0.5)
                      ;;Note carefully the order of the leaves of the network.  There
                      ;;is currently an implicit dependency here on that order, the order
                      ;;of the loss functions and the order of the training and test
                      ;;labels which is probably an argument to specify all of that
                      ;;in the network description
                      (desc/split [[(desc/linear->logistic autoencoder-size
                                                           :l2-max-constraint 2.0)
                                    (desc/linear 784)]
                                   [(desc/linear->softmax 10)]])]
        labels {:training [@training-data @training-labels]
                :test [@test-data @test-labels]}
        loss-fn [(opt/mse-loss) (opt/softmax-loss)]]
    {:network-desc network-desc
     :labels labels
     :loss-fn loss-fn}))

(def network-fns
  {:linear create-linear
   :logistic create-logistic
   :logistic-dropout-full create-dropout-logistic-full
   :multiple-target-dropout create-multi-target-dropout
   :mnist-multiple-target-dropout create-mnist-multi-target})


(defn train-and-evaluate
  [network-params optimiser]
  (resource/with-resource-context
   (let [n-epochs 8
         batch-size 10
         {:keys [network-desc loss-fn labels]} network-params
         network (build-and-create-network network-desc)
         training-labels (get labels :training)
         test-labels (get labels :test)
         network (gpu-train/train network optimiser loss-fn @training-data training-labels
                                  batch-size n-epochs
                                  @test-data test-labels)]
     {:network (vec (flatten (desc/network->description network)))})))


(defonce networks (atom nil))


(defn train-network
  [net-name]
  (with-bindings {#'cudnn/*cudnn-datatype* (float 0.0)}
    (let [network-fn (get network-fns net-name)
          _ (println "training" net-name)
          network-and-score (train-and-evaluate (network-fn) (opt/adam))]
      (swap! networks assoc net-name
             network-and-score))))


(defn train-networks
  []
  (try
    ;;Until we start to make sense of cuda streams, there is no benefit
    ;;of a pmap here and I was getting crashes.
    (doall (map #(train-network %) (keys network-fns)))
    (catch Exception e (do (println "caught error: ") (println  e) (throw e))))
  (keys @networks))

(defn get-trained-network
  [net-name]
  (:network (get @networks net-name)))

(defn check-l2-constraint
  [net-key size]
  (let [network (get-trained-network net-key)
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
  (let [network (get-trained-network net-key)
        first-linear-layer (first (filter #(= :linear (get % :type))
                                          network))

        weights (:weights first-linear-layer)
        filter-count (long (m/row-count weights))
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
  (let [network (get-trained-network net-key)
        network-and-next (map vector network (concat (list  {}) network))
        encoder (remove #(= :dropout (:type %))
                        (mapv first (take-while #(not= (get (second %) :type) :logistic)
                                                network-and-next)))
        encoder (vec (flatten [(desc/input 28 28 1)
                               encoder]))
        encoder-net (desc/build-and-create-network encoder)
        results (net/run encoder-net (vec (take sample-count @test-data)))]
    results))

(defn plot-network-tsne
  [net-key & {:keys [sample-count iterations]
              :or {sample-count 10000
                   iterations 1000}}]
  (let [_ (println "running network" net-key)
        samples (run-network net-key)
        labels (vec (take sample-count @test-labels))
        _ (println (format "tsne-ifying %d samples" sample-count))
        tsne-data (tsne/tsne samples 2 :iters iterations)]
    (plot-tsne-data tsne-data labels (name net-key))))
