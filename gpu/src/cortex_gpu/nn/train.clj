(ns cortex-gpu.nn.train
  (:require [cortex-gpu.nn.cudnn :as cudnn]
            [cortex.nn.protocols :as cp]
            [cortex.dataset :as ds]
            [cortex-gpu.nn.layers :as layers]
            [cortex-gpu.nn.batch :as batch]
            [clojure.core.matrix :as m]
            [cortex.nn.network :as net]
            [cortex-gpu.nn.description :as gpu-desc]
            [cortex.nn.description :as desc]
            [cortex-gpu.cuda :as cuda]
            [cortex-gpu.resource :as resource]
            [cortex-gpu.util :refer [get-or-allocate] :as util]
            [cortex-gpu.optimise :as opt]
            [cortex.optimise :as cortex-opt]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* true)

(defn make-training-config
  ([network optimiser loss-fn
    batch-size epoch-count
    dataset & [cv-dataset]]
   {:network network :optimiser optimiser :loss-fn loss-fn
    :batch-size batch-size :epoch-count epoch-count
    :dataset dataset :cv-dataset cv-dataset}))


(defn train-step
  [{:keys [network loss-fn] :as train-config} input answer]
  (let [network (cp/multi-forward network input)
        output (cp/multi-output network)
        loss-fn (mapv #(opt/calculate-loss-gradient %1 %2 %3)
                      loss-fn output answer)
        loss-gradient (mapv opt/loss-gradient loss-fn)
        network (cp/multi-backward network input loss-gradient)]
    (assoc train-config :network network :loss-fn loss-fn)))


(defn optimise
  [{:keys [network optimiser batch-size] :as train-config}]
  (let [gradients (layers/gradients network)
        parameters (layers/parameters network)
        optimiser (opt/compute-parameters! optimiser gradients
                                           parameters batch-size)]
    (util/zero-many gradients)
    (layers/post-update network)
    (assoc train-config
           :network network
           :optimiser optimiser)))


(defn train-epoch
  [train-config]
  (let [train-config (batch/setup-batching-system-per-epoch train-config :training)
        batch-count (batch/get-num-batches train-config)]
    (reduce (fn [train-config idx]
              (let [{:keys [train-config input-buffers output-buffers]}
                    (batch/get-batching-system-buffers train-config idx :training)]
               (-> train-config
                   (train-step input-buffers output-buffers)
                   optimise)))
            train-config
            (range batch-count))))


(defn run-setup-network
  "Produces a persistent vector of double arrays"
  [{:keys [network batch-size] :as train-config} batch-type]
  (let [train-config (batch/setup-batching-system-per-epoch train-config batch-type)
        next-output (fn [train-config idx]
                      (let [{:keys [train-config input-buffers]} (batch/get-batching-system-buffers train-config idx batch-type)]
                        (cp/multi-output (cp/multi-calc network input-buffers))))
        n-output (cp/multi-output-size network)
        num-batches (batch/get-num-batches train-config)]
    (if (= num-batches 0)
      []
      (let [[train-config results] (reduce
                                    (fn [[train-config retval] idx]
                                      (let [output (next-output train-config idx)]
                                        [train-config
                                         (conj retval
                                               (if (sequential? output)
                                                 (mapv cudnn/to-double-array output)
                                                 (cudnn/to-double-array output)))]))
                                    [train-config []]
                                    (range num-batches))
            result-processor (fn [results n-output]
                               (vec (mapcat #(map vec (partition n-output (seq %))) results)))]
        ;;De-interleave the results
        (mapv (fn [n-output idx]
                (let [nth-results (map #(nth % idx) results)]
                  (result-processor nth-results n-output)))
              n-output
              (range (count n-output)))))))


(defn average-loss
  "cpu only function to evaluate a network.  Should be moved to cortex."
  [loss-fn guesses answers]
  (let[num-guesses (first (m/shape guesses))
       num-labels (first (m/shape answers))
       _ (when-not (= num-guesses num-labels)
           (throw (Exception. (format "Number of guesses %d and number of labels %d mismatch"
                                      num-guesses num-labels))))
       loss-items (map #(cp/loss loss-fn %1 %2)
                       (m/rows guesses) (m/rows answers))
       aggregate-loss (double (reduce + loss-items))]
    (/ aggregate-loss num-guesses)))


(defn evaluate-training-network
  "Run the network and return the average loss across all cv-input"
  [{:keys [network loss-fn batch-size] :as train-config}]
  (let [guesses (run-setup-network train-config :testing)
        cpu-labels (batch/get-cpu-labels train-config :testing)]
    (mapv average-loss loss-fn guesses cpu-labels)))


(defn println-report-epoch
  [epoch-idx train-config]
  (if (batch/has-cpu-labels? train-config :testing)
    (println (format "Epoch loss: %s"
                     (evaluate-training-network train-config)))
    (println (format "Epoch %d finished" epoch-idx))))


(def ^:dynamic *train-epoch-reporter* println-report-epoch)


(defn run-train-optimise-loop
  [{:keys [epoch-count batch-size] :as train-config}]
   (reduce (fn [train-config epoch]
             (let [train-config (train-epoch train-config)]
               ;;Use simple println reporting by default
               (when *train-epoch-reporter*
                 (*train-epoch-reporter* epoch train-config))
               train-config))
           (batch/setup-batching-system train-config :training)
           (range epoch-count)))


(defn train
  [network optimiser loss-fn
   training-data training-labels
   batch-size n-epochs
   & [cv-data cv-labels]]
  (let [cv-dataset (when (and cv-data cv-labels)
                     [cv-data cv-labels])
        loss-fn (if-not (sequential? loss-fn)
                  [loss-fn]
                  loss-fn)
        train-config (make-training-config network optimiser loss-fn
                                           batch-size n-epochs
                                           [training-data training-labels]
                                           cv-dataset)]
    (resource/with-resource-context
     (-> (assoc train-config :batching-system (batch/->OnGPUBatchingSystem))
         (update-in [:network] #(cp/setup % (:batch-size train-config)))
         (run-train-optimise-loop)
         (get :network)))))


(defn run
  [network data & {:keys [batch-size]
                   :or {batch-size 10}}]
  (resource/with-resource-context
    (as-> {:network network :batch-size batch-size :dataset [data]
           :batching-system (batch/->OnGPUBatchingSystem)} train-config
      (batch/setup-batching-system train-config :running)
      (update-in train-config [:network] #(cp/setup % (:batch-size train-config)))
      (run-setup-network train-config :running))))

(defn evaluate-softmax
  [network data labels]
  (let [net-run-results (first (run network data))
        results-answer-seq (mapv vector
                                 (net/softmax-results-to-unit-vectors net-run-results)
                                 labels)
        correct (count (filter #(m/equals (first %) (second %)) results-answer-seq))]
    (double (/ correct (count results-answer-seq)))))

(defn mse
  [guesses answers]
  (average-loss (cortex-opt/mse-loss) (first guesses) answers))


(defn evaluate-mse
  [network data labels]
  (let [net-run-results (run network data)]
    (mse net-run-results labels)))


(defn train-next
  [network-desc optimiser loss-fn dataset batch-size n-epochs input-labels output-labels]
  (resource/with-resource-context
    (let [network (gpu-desc/build-and-create-network network-desc)
          loss-fn (if-not (sequential? loss-fn)
                    [loss-fn]
                    loss-fn)
          train-config (make-training-config network optimiser loss-fn
                                             batch-size n-epochs dataset)
          dataset-names (map :name (ds/shapes dataset))
          index-names (map-indexed vector dataset-names)
          find-label-fn (fn [label]
                              (ffirst (filter (fn [[idx name]]
                                                (= label name))
                                              index-names)))
          input-indexes (mapv find-label-fn input-labels)
          output-indexes (mapv find-label-fn output-labels)
          batching-system (batch/->DatasetBatchingSystem input-indexes output-indexes batch-size)]
      (-> (assoc train-config :batching-system batching-system)
          (update-in [:network] #(cp/setup % (:batch-size train-config)))
          (run-train-optimise-loop)
          (get :network)
          (desc/network->description)))))
