(ns cortex.experiment.tensorboard-test
  (:require [clojure.java.io :as io]
            [clojure.string :as s]
            [clojure.test :refer :all]
            [cortex.loss :as loss]
            [cortex.nn.layers :as layers]
            [tfevent-sink.event-io :as eio]
            [cortex.nn.execute :as execute]
            [cortex.experiment.train :as train]
            [cortex.nn.network :as network]
            [clojure.core.async :as async :refer [>! <! <!! >!! chan close! go alts!]]
            [cortex.metrics :as met]
            [cortex.experiment.logistic-test :as lt]))

;; "Default" dataset taken from: https://cran.r-project.org/web/packages/ISLR/index.html
;; Distributed under the GPL-2 LICENSE
;; The format is USER_ID,STUDENT?,DEFAULT?,BALANCE,INCOME

(def default-dataset lt/default-dataset)

(def linear-with-batch-norm
  [(layers/input 2 1 1 :id :data)
   (layers/batch-normalization)
   ;;Fix the weights to make the unit test work.
   (layers/linear 1 :weights [[-0.2 0.2]])
   (layers/logistic :id :labels)])

(def linear-without-batch-norm
  [(layers/input 2 1 1 :id :data)
   ;;Fix the weights to make the unit test work.
   (layers/linear 1 :weights [[-0.2 0.2]])
   (layers/logistic :id :labels)])

(def relu-with-batch-norm
  [(layers/input 2 1 1 :id :data)
   (layers/batch-normalization)
   ;;Fix the weights to make the unit test work.
   (layers/linear->relu 1)
   (layers/logistic :id :labels)])

(def tanh-with-batch-norm
  [(layers/input 2 1 1 :id :data)
   (layers/batch-normalization)
   ;;Fix the weights to make the unit test work.
   (layers/linear->tanh 1)
   (layers/logistic :id :labels)])

(defn get-metric
  [metric-fn actl-label pred-label]
  (let [lfn #(mapv (comp first :labels) %)
        pred-lab (->> pred-label lfn (mapv #(if (> % 0.5) 1.0 0.0)))]
    (metric-fn (lfn actl-label) pred-lab)))

(defn log-eval-metrics
  "Given the context, old network, the new network and a test dataset,
  return a map indicating if the new network is indeed the best one
  and the network with enough information added to make comparing
  networks possible.
    {:best-network? boolean
     :network (assoc new-network :whatever information-needed-to-compare).}"
  [metrics-chan
   {:keys [batch-size context]}
   {:keys [new-network old-network test-ds train-ds]}] ;;change per epoch
  (let [batch-size (long batch-size)
        get-label (fn [dset] (execute/run new-network dset
                                          :batch-size batch-size
                                          :loss-outputs? true
                                          :context context))
        labels (get-label test-ds)
        predicted-train-ds-labels (get-label train-ds)
        loss-on (fn [dset] (execute/execute-loss-fn new-network labels dset))
        loss-fn (loss-on test-ds)
        loss-val (apply + (map :value loss-fn))
        train-loss (apply + (map :value (loss-on train-ds)))
        epoch-count (get new-network :epoch-count)
        bcast-msg (fn []
                    (assoc {}
                           :train-loss train-loss :test-loss loss-val
                           :train-accuracy (get-metric met/accuracy train-ds predicted-train-ds-labels)
                           :test-accuracy (get-metric met/accuracy test-ds labels)
                           :train-precision (get-metric met/precision train-ds predicted-train-ds-labels)
                           :test-precision (get-metric met/precision test-ds labels)
                           :train-recall (get-metric met/recall train-ds predicted-train-ds-labels)
                           :test-recall (get-metric met/recall test-ds labels)))
        _ (go (>! metrics-chan (bcast-msg)))
        current-best-loss (if-let [best-loss (get old-network :cv-loss)]
                            ;; TODO: Is there a bug here? What if the best-loss isn't sequential?
                            (when (sequential? best-loss)
                              (apply + (map :value best-loss))))
        best-network? (or (nil? current-best-loss)
                          (< (double loss-val)
                             (double current-best-loss)))]
    ;(println (format "Loss for epoch %s: %s" (get new-network :epoch-count) loss-val))
    {:best-network? best-network?
     :network (assoc new-network :cv-loss loss-fn)}))

(defn- spit-events
  [file-path metrics]
  (let [ev (mapv eio/make-event
                 (mapv (partial str "metrics/") (mapv name (keys metrics)))
                 (vals metrics))]
    (eio/append-events file-path ev)))

(defn- start-tensorboard-broadcast
  [file-path channel num-msgs]
  (do
    (eio/create-event-stream file-path)
    (go (dotimes [n num-msgs]
          (let [[v _] (alts! [channel])]
            (spit-events file-path v))))))

(defn- train-and-eval
  [train-ds test-ds network-description log-file]
  (let [epoch-count 20
        metrics-chan (chan)
        event-writer-chan (start-tensorboard-broadcast log-file metrics-chan epoch-count)
        _ (train/train-n network-description train-ds test-ds
                         :batch-size 50 :epoch-count epoch-count
                         :test-fn (partial log-eval-metrics metrics-chan))]
    ;;wait for the listener to finish reading epoch-count events
    (<!! event-writer-chan)))

(deftest test-listener
  (let [ds (shuffle (default-dataset))
          log-path "/tmp/tflogs/"
          ds-count (count ds)
          train-ds (take (int (* 0.9 ds-count)) ds)
          test-ds (drop (int (* 0.9 ds-count)) ds)
          train-fn (partial train-and-eval train-ds test-ds)]
      (mapv train-fn [linear-with-batch-norm
                      linear-without-batch-norm relu-with-batch-norm
                      tanh-with-batch-norm]
            (mapv #(let [path (str log-path % "/tfevents." % ".out")]
                     (io/make-parents path)
                     path)
                  ["linear"
                   "linear_nobatchnorm" "relu" "tanh"]))
      (is (.exists (io/file (str log-path "/linear/tfevents.linear.out"))))))
