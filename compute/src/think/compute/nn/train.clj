(ns think.compute.nn.train
  (:require [think.compute.driver :as drv]
            [think.compute.math :as math]
            [think.compute.nn.backend :as nn-backend]
            [think.compute.nn.layers :as layers]
            [think.compute.optimise :as opt]
            [think.compute.math :as math]
            [think.compute.batching-system :as batch]
            [think.datatype.core :as dtype]
            [think.resource.core :as resource]
            [cortex.nn.protocols :as cp]
            [cortex.nn.description :as desc]
            [cortex.dataset :as ds]
            [think.compute.nn.description :as compute-desc]
            [clojure.core.matrix :as m]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* true)


(defn train-step
  [{:keys [network loss-fn] :as train-config} input answer]
  (let [network (cp/prepare-forward network)
        network (cp/multi-forward network input)
        output (cp/multi-output network)
        loss-fn (mapv #(opt/calculate-loss-gradient %1 %2 %3)
                      loss-fn output answer)
        loss-gradient (mapv opt/loss-gradient loss-fn)
        network (cp/multi-backward network input loss-gradient)]
    (assoc train-config :network network :loss-fn loss-fn)))


(defn optimise
  [{:keys [network optimiser] :as train-config}]
  (let [backend (layers/get-backend network)
        gradients (layers/gradients network)
        parameters (layers/parameters network)
        learning-attenuation (layers/learning-attenuation network)
        batch-size (long (layers/batch-size network))
        alpha (/ 1.0 batch-size)
        optimiser (opt/batch-update optimiser)]
    (reduce (fn [offset [gradients parameters learning-attenuation]]
              (opt/compute-parameters! optimiser (* learning-attenuation alpha)
                                       offset gradients parameters)
              (+ ^long offset ^long (math/ecount parameters)))
            0
            (partition 3 (interleave gradients parameters learning-attenuation)))
    (doseq [grad gradients]
      (drv/memset (drv/get-stream backend) (math/device-buffer grad) 0 0 (math/ecount grad)))
    (layers/post-update network)
    (assoc train-config
           :network network
           :optimiser optimiser)))

(defn- train-batches
  [{:keys [batching-system] :as train-config}]
  (reduce (fn [train-config {:keys [input-buffers output-buffers]}]
            (-> (train-step train-config input-buffers output-buffers)
                optimise))
          train-config
          (batch/get-batches batching-system :training true)))


(defn host-buffer->double-array
  [host-buffer]
  (let [retval (double-array (m/ecount host-buffer))]
    (dtype/copy! host-buffer 0 retval 0 (m/ecount host-buffer))
    retval))


(defn- recur-run-config
  [{:keys [network batching-system] :as train-config} batch-sequence]
  (when-let [next-batch (first batch-sequence)]
    (let [backend (layers/get-backend network)
          {:keys [input-buffers]} next-batch
          ;;Note no prepare-calc call
          network (cp/multi-calc network input-buffers)
          train-config (assoc train-config :network network)
          n-output-vec (cp/multi-output-size (:network train-config))
          splitter (fn [buffer-seq]
                     (map (fn [^long n-output output-buffer]
                            ;;separate into one output per batch
                            (let [n-batches (long (quot (long (m/ecount output-buffer))
                                                        (long n-output)))]
                              ;;!!!!This absolutely *cannot* be lazy!!!
                              (mapv (fn [^long batch-idx]
                                      (let [output-data (double-array n-output)]
                                        (dtype/copy! output-buffer (* batch-idx n-output)
                                                     output-data 0 n-output)
                                        output-data))
                                    (range n-batches))))
                          n-output-vec buffer-seq))]
      (cons {:train-config train-config
             :inferences (splitter (map #(nn-backend/to-double-array backend %)
                                        (cp/multi-output network)))}
            (lazy-seq (recur-run-config train-config (rest batch-sequence)))))))


(defn- run-config
  "Returns a map of {train-config inferences labels} from the outputs
of the network for each"
  [{:keys [network batching-system] :as train-config} batch-type]
  (let [eval-sequence
        (lazy-seq (recur-run-config train-config
                                    ;;false here because we don't need the result
                                    ;;uploaded to the gpu
                                    (batch/get-batches batching-system batch-type false)))
        n-outputs (count (cp/multi-output-size network))
        coalesced-data (reduce (fn [coalesced-data {:keys [train-config inferences labels]}]
                                 (assoc coalesced-data
                                        :train-config train-config
                                        :inferences (conj (:inferences coalesced-data)
                                                          (vec inferences))))
                               {:inferences []}
                               eval-sequence)]
    {:train-config (:train-config coalesced-data)
     :inferences (ds/batches->columns (:inferences coalesced-data))}))


(defn evaluate-training-network
  "Run the network and return the average loss across all cv-input"
  [labels train-config batch-type]
  (let [{:keys [train-config inferences] :as run-config-output}
        (run-config train-config batch-type)
        {:keys [loss-fn]} train-config]
    ;;when there were any batches to begin with
    (when (count (first inferences))
      (assoc run-config-output
             :avg-loss (mapv opt/average-loss loss-fn inferences labels)))))


(defn println-report-epoch
  [labels epoch-idx {:keys [batching-system dataset] :as train-config}]
  (if-let [evaluated-network-data (evaluate-training-network labels train-config
                                                             :cross-validation)]
    (let [{:keys [train-config avg-loss]} evaluated-network-data]
      (println (format "Epoch loss: %s" avg-loss))
      train-config)
    (do
      (println (format "Epoch %d finished" epoch-idx))
      train-config)))


(defn train-epoch-seq
  "Infinite sequence of train configs, one for each epoch."
  [train-config]
  (cons train-config (lazy-seq (train-epoch-seq (train-batches train-config)))))


(defn build-train-config
  [net optimiser dataset input-labels output-labels-and-loss]
  (let [backend (layers/get-backend net)
        batch-size (layers/batch-size net)
        batching-system (-> (batch/create-dataset-batching-system
                             input-labels (mapv first output-labels-and-loss) batch-size
                             dataset (drv/get-driver backend) (drv/get-stream backend)
                             (dtype/get-datatype backend))
                            batch/setup)
        loss-fns (mapv (fn [[label loss] output-size]
                         (opt/setup-loss loss backend batch-size output-size))
                       output-labels-and-loss (cp/multi-output-size net))
        optimiser (opt/setup-optimiser optimiser backend (layers/parameter-count net))]
    {:network net :optimiser optimiser :loss-fn loss-fns :batching-system batching-system}))


(defn create-train-epoch-sequence
  "Create an infinite sequence of train-configs where each next config
is trained one more epoch than the config before it.
Drops the initial config (which could be completely untrained).  Note that
this could allocate a significant amount of resources so you cannot pass this
sequence as is around a program."
  [net optimiser dataset input-labels output-labels-and-loss]
  (->> (build-train-config net optimiser dataset input-labels output-labels-and-loss)
       train-epoch-seq
       (drop 1)))


(defn train
  "Epoch train filter takes an epoch-index and a train config and produces a new
  train config; providing an opportunity for side effects (e.g., printing)."
  [net optimiser dataset input-labels output-labels-and-loss epoch-count
   & {:keys [epoch-train-filter]
      :or {epoch-train-filter :unset}}]
  (resource/with-resource-context
    (let [epoch-filter (if epoch-train-filter
                         (if (= epoch-train-filter :unset)
                           (let [output-labels (mapv first output-labels-and-loss)
                                 dataset-labels (ds/batches->columns
                                                 (ds/get-batches dataset (layers/batch-size net)
                                                               :cross-validation output-labels))]
                             (partial println-report-epoch dataset-labels))
                           epoch-train-filter)
                         (fn [idx train-config]
                           train-config))]
      (->> (create-train-epoch-sequence net optimiser dataset input-labels
                                        output-labels-and-loss)
           (map-indexed epoch-filter)
           (take epoch-count)
           last
           :network))))


(defn infer-network
  [net dataset input-labels output-labels & {:keys [batch-type]
                                             :or {batch-type :holdout}}]
  (resource/with-resource-context
    (let [backend (layers/get-backend net)
          batch-size (layers/batch-size net)
          batching-system (-> (batch/create-dataset-batching-system input-labels output-labels
                                                                    batch-size dataset
                                                                    (drv/get-driver backend)
                                                                    (drv/get-stream backend)
                                                                    (dtype/get-datatype backend))
                              batch/setup)]
      (select-keys (run-config {:network net :batching-system batching-system} batch-type)
                   [:inferences :labels]))))


(defn run
  "Run a network products a vector of output sequences, one sequence for
each output of the network."
  [net dataset input-labels & {:keys [batch-type]
                               :or {batch-type :holdout}}]
  (:inferences (infer-network net dataset input-labels [] :batch-type batch-type)))
