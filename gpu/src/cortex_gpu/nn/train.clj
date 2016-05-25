(ns cortex-gpu.nn.train
  (:require [cortex-gpu.nn.cudnn :as cudnn]
            [cortex.nn.protocols :as cp]
            [cortex-gpu.nn.layers :as layers]
            [cortex-gpu.nn.batch :as batch]
            [clojure.core.matrix :as m]
            [cortex.nn.network :as net]
            [cortex-gpu.cuda :as cuda]
            [cortex-gpu.resource :as resource]
            [cortex-gpu.util :refer [get-or-allocate] :as util]
            [cortex-gpu.optimise :as opt]
            [cortex.optimise :as cortex-opt]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* true)

(defn make-training-config
  [network optimiser loss-fn
   batch-size epoch-count
   dataset cv-dataset]
  {:network network :optimiser optimiser :loss-fn loss-fn
   :batch-size batch-size :epoch-count epoch-count
   :dataset dataset :cv-dataset cv-dataset})


(defn train-step
  [{:keys [network loss-fn] :as train-config} input answer]
  (let [network (cp/forward network input)
        output (cp/output network)
        loss-fn
        (if (sequential? loss-fn)
          (mapv #(opt/calculate-loss-gradient %1 %2 %3)
                loss-fn output answer)
          (opt/calculate-loss-gradient loss-fn output (first answer)))
        loss-gradient (if (sequential? loss-fn)
                        (mapv opt/loss-gradient loss-fn)
                        (opt/loss-gradient loss-fn))
        network (cp/backward network input loss-gradient)]
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


(defn run-setup-network
  "Produces a persistent vector of double arrays"
  [{:keys [network batch-size gpu-dataset] :as train-config}]
  (let [gpu-data (first gpu-dataset)
        n-output (layers/output-size network)
        data-shape (cudnn/shape gpu-data)
        [^long n-rows ^long n-cols] data-shape
        train-config (batch/upload-sequential-indexes train-config n-rows)
        num-batches (quot n-rows batch-size)
        batch-buffer (get train-config batch/data-keyword)
        next-output (fn [train-config idx]
                      (let [train-config (batch/load-batch-buffer train-config gpu-data
                                                                  idx batch-buffer)]
                        (cp/output (cp/calc network batch-buffer))))
        n-output (layers/output-size network)]
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
        (if (sequential? n-output)
          ;;De-interleave the results
          (mapv (fn [n-output idx]
                  (let [nth-results (map #(nth % idx) results)]
                    (result-processor nth-results n-output)))
                n-output
                (range (count n-output)))
          (result-processor results n-output))))))


(defn average-loss
  "cpu only function to evaluate a network.  Should be moved to cortex."
  [loss-fn guesses answers]
  (let[num-guesses (count guesses)
       num-labels (count answers)
       _ (when-not (= num-guesses num-labels)
           (throw (Exception. (format "Number of guesses %d and number of labels %d mismatch"
                                      num-guesses num-labels))))
       loss-items (map #(cp/loss loss-fn %1 %2)
                       guesses answers)
       aggregate-loss (double (reduce + loss-items))]
    (/ aggregate-loss num-guesses)))


(defn evaluate-training-network
  "Run the network and return the average loss across all cv-input"
  [{:keys [network loss-fn gpu-cv-dataset batch-size] :as train-config}]
  (let [[gpu-data cpu-labels] gpu-cv-dataset
        guesses (run-setup-network (assoc train-config :gpu-dataset [gpu-data]))]
    (if (sequential? loss-fn)
      (mapv average-loss loss-fn guesses cpu-labels)
      (average-loss loss-fn guesses cpu-labels))))


(defn train-epoch
  [{:keys [gpu-dataset ^long batch-size] :as train-config}]
  (let [[data labels] gpu-dataset
        [^long total-input-count n-input] (cudnn/shape data)
        num-batches (quot total-input-count batch-size)
        train-config (batch/upload-randomized-indexes train-config total-input-count)
        input-buffer (get train-config batch/data-keyword)
        output-buffer (get train-config batch/label-keyword)]
    (reduce (fn [train-config idx]
              (-> train-config
                  (batch/load-batch-buffer data idx input-buffer)
                  (batch/load-batch-buffer labels idx output-buffer)
                  (train-step input-buffer output-buffer)
                  optimise))
            train-config
            (range num-batches))))

(defn println-report-epoch
  [epoch-idx {:keys [gpu-cv-dataset] :as train-config}]
  (if-let [cv-cpu-labels (second gpu-cv-dataset)]
    (println (format "Epoch loss: %s"
                     (evaluate-training-network train-config)))
    (println (format "Epoch %d finished" epoch-idx))))


(def ^:dynamic *train-epoch-reporter* println-report-epoch)



(defn run-train-optimise-loop
  [{:keys [gpu-dataset gpu-cv-dataset epoch-count] :as train-config}]
   (let [cv-cpu-labels (when gpu-cv-dataset
                        (second gpu-cv-dataset))]
    (reduce (fn [train-config epoch]
              (let [train-config (train-epoch train-config)]
                ;;Use simple println reporting by default
                (when *train-epoch-reporter*
                  (*train-epoch-reporter* epoch train-config))
                train-config))
            train-config
            (range epoch-count))))

(defn setup-train-config
  [train-config]
  (update-in train-config [:network] #(layers/setup % (:batch-size train-config))))


(defn upload-datasets
  "Upload the dataset to the gpu if it is not already uploaded"
  [{:keys [dataset cv-dataset gpu-dataset] :as train-config}]
  (if gpu-dataset
    train-config
    (let [_ (println "Loading dataset to gpu")
          gpu-dataset (batch/load-dataset-to-gpu dataset)
          gpu-cv-data (when cv-dataset
                        (batch/load-dataset-to-gpu (take 1 cv-dataset)))
          train-cv-dataset (when cv-dataset
                             [(first gpu-cv-data) (second cv-dataset)])]
      (assoc train-config :gpu-dataset gpu-dataset :gpu-cv-dataset train-cv-dataset))))




(defn train-uploaded-train-config
  [{:keys [gpu-dataset batch-size gpu-cv-dataset] :as train-config}]
  (let [[data labels] gpu-dataset
        [total-input-count n-input] (cudnn/shape data)
        ;;Labels is a vector of label matrixes so the outputs are a vector
        ;;of output sizes
        n-output-vec (map (comp second cudnn/shape) labels)
        batch-count (quot total-input-count batch-size)]
    (-> train-config
        (batch/setup batch-size total-input-count n-input n-output-vec)
        setup-train-config
        run-train-optimise-loop)))

(defn verify-dataset-data-label-counts
  [[data labels] batch-size network]
  (let [[input-count n-inputs] (m/shape data)
        net-input-size (layers/input-size network)
        [net-output-sizes label-counts-and-output-sizes]
        (if (= 3 (count (m/shape labels)))
          [(layers/output-size network)
           (mapv m/shape labels)]
          [[(layers/output-size network)]
           [(m/shape labels)]])
        label-counts (mapv first label-counts-and-output-sizes)
        output-sizes (mapv second label-counts-and-output-sizes)]

    (when-not (= (long n-inputs) (long net-input-size))
      (throw (Exception.
              (format "Network input size does not match data column length: %s %s"
                      net-input-size n-inputs))))
    (when-not (= 0 (rem input-count batch-size))
      (throw (Exception. "Input count is not evenly divisible by batch size.")))
    (when-not (= (count net-output-sizes)
                 (count label-counts-and-output-sizes))
      (throw (Exception. (format "Network output count differs from label count: %s %s"
                                 (count net-output-sizes)
                                 (count label-counts-and-output-sizes)))))
    (when-not (seq (filter #(apply = %) (map list output-sizes net-output-sizes)))
      (throw (Exception. (format "Network output size and output vec mismatch %s %s"
                                 output-sizes net-output-sizes))))
    (when-not (apply = label-counts)
      (throw (Exception. (format "Label counts differ: %s" label-counts))))
    (when-not (= (first label-counts) input-count)
      (throw (Exception. (format "Input count differs from output count: %s %s"
                                 input-count (first label-counts)))))))


(defn verify-incoming-dataset
  "Verify the dataset sizes are sane and match then network config"
  [{:keys [dataset batch-size cv-dataset network] :as train-config}]
  (verify-dataset-data-label-counts dataset batch-size network)
  (when cv-dataset
    (verify-dataset-data-label-counts cv-dataset batch-size network)))


(defn run-train-optimise
  [train-config]
  (verify-incoming-dataset train-config)
  (resource/with-resource-context
    (let [train-config (upload-datasets train-config)]
      (println "training")
      (train-uploaded-train-config train-config))))


(defn train-pre-uploaded
  [network optimiser loss-fn batch-size n-epochs {:keys [dataset cv-dataset] :as train-config}]
  (-> (merge train-config (make-training-config network optimiser loss-fn
                                                batch-size n-epochs
                                                dataset cv-dataset))
      train-uploaded-train-config
      :network))


(defn train
  [network optimiser loss-fn
   training-data training-labels
   batch-size n-epochs
   & [cv-data cv-labels]]
  (let [cv-dataset (when (and cv-data cv-labels)
                     [cv-data cv-labels])
        train-config (make-training-config network optimiser loss-fn
                                           batch-size n-epochs
                                           [training-data training-labels]
                                           cv-dataset)
        train-config (run-train-optimise train-config)]
    (:network train-config)))


(defn run
  [network data & {:keys [batch-size]
                   :or {batch-size 10}}]
  (resource/with-resource-context
    (let [gpu-data (cudnn/array data)
          [total-input-count n-input] (m/shape data)
          batch-count (quot total-input-count batch-size)
          train-config {:network network :batch-size batch-size :gpu-dataset [gpu-data]}]
      (-> train-config
          (batch/setup batch-size total-input-count
                       n-input
                       (layers/output-size network))
          setup-train-config
          run-setup-network ))))


(defn evaluate-softmax
  [network data labels]
  (let [net-run-results (run network data)
        results-answer-seq (mapv vector
                                 (net/softmax-results-to-unit-vectors net-run-results)
                                 labels)
        correct (count (filter #(m/equals (first %) (second %)) results-answer-seq))]
    (double (/ correct (count results-answer-seq)))))


(defn mse
  [guesses answers]
  (average-loss (cortex-opt/mse-loss) guesses answers))


(defn evaluate-mse
  [network data labels]
  (let [net-run-results (run network data)]
    (mse net-run-results labels)))
