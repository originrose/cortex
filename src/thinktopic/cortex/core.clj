(ns thinktopic.cortex.core
  (:gen-class)
  (:use
    [thinktopic.cortex.lab core charts]
    [clojure.core.matrix])
  (:require
    [thinktopic.cortex.gui :as viz]
    [task.core :as task]
    [mikera.vectorz.core :as vectorz]
    [thinktopic.datasets.mnist :as mnist])
  (:import [mikera.vectorz Op Ops])
  (:import [mikera.vectorz.ops ScaledLogistic Logistic Tanh])
  (:import [nuroko.coders CharCoder])
  (:import [mikera.vectorz AVector Vectorz]))


; TODO: probably a faster way to do this by generating a random sparse vector
(defn denoising-task
  [data noise-ratio]
  (let [ins (vec data)]
    (proxy [nuroko.task.ExampleTask] [ins ins]
      (getInput [out-vec]
        (proxy-super getInput out-vec)
        (dotimes [i (ecount out-vec)]
          (when (> (rand) noise-ratio)
            (vectorz/set out-vec i 0.0)))))))


(defn autoencode-task
  [data & [{:keys [noise-pct]}]]
  (if (pos? noise-pct)
    ; Autoencode with corrupted input, forcing the network to filter out the noise
    (denoising-task data noise-pct)
    ; Pure autoencoding task
    (identity-task data)))


(defn encoder
  [{:keys [input-size hidden-size max-weights encoder-activation
           sparsity-weight sparsity-target dropout] :as config}]
  (stack
    (offset :length input-size :delta -0.5)
    (neural-network :inputs input-size
                    :outputs hidden-size
                    :layers 1
                    :max-weight-length max-weights
                    :output-op encoder-activation
                    :dropout dropout
                    )
    (sparsifier :length hidden-size :weight sparsity-weight :target-mean sparsity-target)))


(defn decoder
  [{:keys [input-size hidden-size max-weights decoder-activation] :as config}]
  (stack
    (offset :length hidden-size :delta -0.5)
    (neural-network :inputs hidden-size
                    :outputs input-size
    ;                :max-weight-length max-weights
                    :output-op decoder-activation
                    :layers 1)))


(defn autoencoder
  [{:keys [train-data noise-pct] :as config}]
  (let [enco    (encoder config)
        deco    (decoder config)
        model   (connect enco deco)
        task    (autoencode-task train-data config)
        trainer (supervised-trainer model task)]
    {:task task
     :encoder enco
     :decoder deco
     :model model
     :trainer trainer}))


(defn classifier
  [{:keys [classes train-labels train-data classifier-size] :as config} ae]
  (let [out-coder (class-coder :values classes)
        task (mapping-task
               (apply hash-map
                      (interleave train-data train-labels))
               :output-coder out-coder)
        classifier (stack
                     ;(offset :length classifier-size :delta -0.5)
                     (neural-network :inputs classifier-size
                                     :output-op Ops/LOGISTIC
                                     :outputs (count classes)
                                     :layers 2))
        model   (connect (:encoder ae) classifier)
        trainer (supervised-trainer model task
                   ;;:loss-function nuroko.module.loss.CrossEntropyLoss/INSTANCE
                )]
    {:autoencoder ae
     :task task
     :classifier classifier
     :model model
     :trainer trainer
     :output-coder out-coder}))


(defn classify
  [classifier input-sample decoder]
  (->> input-sample
       (think classifier)
       (decode decoder)))


(defn evaluator-task
  [{:keys [test-data test-labels] :as config} {:keys [output-coder] :as classifier}]
  (mapping-task (apply hash-map
                       (interleave test-data test-labels))
                :output-coder output-coder))


(defn view-samples
  [n data]
  (viz/show (map viz/img (take n data))
        :title (format "First %d samples" n)))


(defn track-classification-error
  "Show chart of training error (blue) and test error (red)"
  [model classify-task test-task]
  (viz/show (viz/time-chart [#(evaluate-classifier classify-task model)
                     #(evaluate-classifier test-task model)]
                    :y-max 1.0)
        :title "Error rate"))


#_(defn show-results
  "Show classification results, errors are starred"
  []
  (viz/show (map (fn [l r] (if (= l r) l (str r "*")))
             (take 100 labels)
             (map recognise (take 100 data)))
        :title "Classification results"))


(defn show-class-separation
  [{:keys [test-data test-labels] :as config} classifier n]
  (viz/show (viz/class-separation-chart classifier
                                (take n test-data)
                                (take n test-labels))))


(defn feature-img
  [vector]
  ((viz/image-generator :width 28
                        :height 28
                        :colour-function viz/weight-colour-rgb)
   vector))


(defn show-reconstructions
  [model data n]
  (viz/show
    (->> (take n data)
         (map (partial think model))
         (map #(viz/img % :colour-function viz/weight-colour-rgb)))
    :title (format "%d samples reconstructed" n)))


(defn show-features
  [layer scale]
  (viz/show (map feature-img (viz/feature-maps layer :scale scale))
            :title "Features"))


(defn chain-task [task1 task2]
  (future
    (task/await-task task1)
    (task/run-task task2)))

(defn run-experiment
  [{:keys [n-reconstructions train-data ae-train-epochs mlp-train-epochs learn-rate]
    :as config}]
  (let [ae     (autoencoder config)
        ;mlp    (classifier config ae)
        ;evaluator (evaluator-task config mlp)
        ae-task (task/task {:repeat ae-train-epochs}
                  (do
                    ((:trainer ae) (:model ae) :learn-rate learn-rate)
                    (show-reconstructions (:model ae) train-data n-reconstructions)))

        ;mlp-task (task/task
        ;           (try
        ;             (track-classification-error (:model mlp) (:task mlp) evaluator)
        ;             (task/run {:repeat mlp-train-epochs}
        ;                       ((:trainer mlp) (:model mlp) :learn-rate learn-rate))

        ;             (catch Exception e
        ;               (println "Error: " e)
        ;               (.printStackTrace e))))
        viz-task (task/task {:repeat true :pause 5000}
                            (show-features (:encoder ae) 5))
        ]

    ;(viz/show (viz/network-graph (:model ae) :line-width 2) :title "Autoencoder : MNIST")
    (view-samples n-reconstructions train-data)

    ;(chain-task ae-task mlp-task)
    (task/run-task viz-task)
    (task/run-task ae-task)

    {:autoencoder ae
     :ae-task ae-task
     ;:classifier  mlp
     ;:evaluator   evaluator
     :viz-task viz-task}))

(defn train-more
  [{:keys [n-reconstructions train-data learn-rate]
    :as config}]
  (let [(task/task {:repeat ae-train-epochs}
                  (do
                    ((:trainer ae) (:model ae) :learn-rate learn-rate)
                    (show-reconstructions (:model ae) train-data n-reconstructions)))]))


;; 60,000 training samples, 10,000 test
(def MNIST-CLASSES     (range 10))
(def MNIST-DATA        @mnist/data-store)
(def MNIST-LABELS      @mnist/label-store)
(def MNIST-TEST-DATA   @mnist/test-data-store)
(def MNIST-TEST-LABELS @mnist/test-label-store)


(def config
  {:input-size      784
   :hidden-size     100
   :classifier-size 30
   :max-weights     0.50
   :noise-pct       0.0
   :dropout         0.0
   :sparsity-weight 0.0
   :sparsity-target 0.2
   :encoder-activation Ops/LOGISTIC ; Ops/TANH ; Ops/RECTIFIER
   :decoder-activation Ops/LOGISTIC ;Ops/LOGISTIC ; Ops/LINEAR ; Ops/RECTIFIER
   :learn-rate 0.1

   :classes       MNIST-CLASSES
   :train-data    MNIST-DATA
   :train-labels  MNIST-LABELS
   :test-data     MNIST-TEST-DATA
   :test-labels   MNIST-TEST-LABELS

   :ae-train-epochs     2
   :mlp-train-epochs    1
   :n-reconstructions 36
   })

(def ex (atom nil))

(defn stop
  []
  ;(overtone.at-at/stop-and-reset-pool! TIMERZ)
  (task/stop-all))

(defn start
  []
  (reset! ex (run-experiment config))
  :running)

(defn -main [& args]
  (start))

