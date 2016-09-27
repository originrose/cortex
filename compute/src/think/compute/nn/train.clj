(ns think.compute.nn.train
  (:require [think.compute.driver :as drv]
            [think.compute.math :as math]
            [think.compute.nn.backend :as nn-backend]
            [think.compute.nn.layers :as layers]
            [think.compute.optimise :as opt]
            [think.compute.math :as math]
            [think.compute.batching-system :as batch]
            [think.compute.datatype :as dtype]
            [resource.core :as resource]
            [cortex.nn.protocols :as cp]
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
        batch-size (long (layers/batch-size network))
        alpha (/ 1.0 batch-size)
        optimiser (opt/batch-update optimiser)]
    (reduce (fn [offset [gradients parameters]]
              (opt/compute-parameters! optimiser alpha offset gradients parameters)
              (+ ^long offset ^long (math/ecount parameters)))
            0
            (partition 2 (interleave gradients parameters)))
    (doseq [grad gradients]
      (drv/memset (drv/get-stream backend) (math/device-buffer grad) 0 0 (math/ecount grad)))
    (layers/post-update network)
    (assoc train-config
           :network network
           :optimiser optimiser)))

(defn- train-batches
  [{:keys [batching-system] :as  train-config}]
  (-> (reduce (fn [{:keys [batching-system] :as train-config} batch-idx]
                (let [{:keys [batching-system input-buffers output-buffers]} (batch/get-batch-buffers batching-system batch-idx)]
                  (-> (assoc train-config :batching-system batching-system)
                      (train-step input-buffers output-buffers)
                      optimise)))
              train-config
              (range (batch/get-num-batches batching-system)))))


(defn- run-config
  "Returns [train-config results]"
  [{:keys [network] :as train-config} batch-type]
  (let [train-config (update-in train-config [:batching-system] #(batch/setup-epoch % batch-type))
        backend (layers/get-backend network)]
    (reduce (fn [[{:keys [batching-system network] :as train-config} results] batch-idx]
              (let [{:keys [batching-system input-buffers]} (batch/get-batch-buffers batching-system batch-idx)
                    ;;Note lack of prepare-forward; there is no prepare-calc call
                    network (cp/multi-calc network input-buffers)]
                [(assoc train-config
                        :batching-system batching-system
                        :network network)
                 (conj results (mapv #(nn-backend/to-double-array backend %)
                                     (cp/multi-output network)))]))
            [train-config []]
            (range (batch/get-num-batches (:batching-system train-config))))))

(defn- reshape-run-config-output
  "Given batched blocks of output (one block per batch) transpose it
to a vector of unbatched output per network-output
takes [train-config results] and returns [train-config results]"
  [[train-config results]]
  (let [n-output-vec (cp/multi-output-size (:network train-config))]
    [train-config
     (vec (map-indexed (fn [idx n-output]
                         (vec (mapcat #(map vec (partition n-output (seq %)))
                                      (map #(nth % idx) results))))
                       n-output-vec))]))


(defn- run-and-reshape
  [train-config batch-type]
  (-> train-config
      (run-config batch-type)
      reshape-run-config-output))


(defn evaluate-training-network
  "Run the network and return the average loss across all cv-input"
  [train-config]
  (let [[train-config guesses] (run-and-reshape train-config :testing)
        {:keys [batching-system loss-fn]} train-config
        cpu-labels (batch/get-cpu-labels batching-system)]
    [train-config (mapv opt/average-loss loss-fn guesses cpu-labels)]))


(defn println-report-epoch
  [epoch-idx {:keys [batching-system dataset] :as train-config}]
  (if (batch/has-cpu-labels? batching-system)
    (let [[train-config avg-loss] (evaluate-training-network train-config)]
      (println (format "Epoch loss: %s" avg-loss))
      train-config)
    (do
      (println (format "Epoch %d finished" epoch-idx))
      train-config)))


(def ^:dynamic *train-epoch-reporter* println-report-epoch)


(defn epoch-progress
  [train-config epoch-idx]
  (if *train-epoch-reporter*
    (*train-epoch-reporter* epoch-idx train-config)
    train-config))


(defn train
  [net optimiser dataset input-labels output-labels-and-loss epoch-count]
  (resource/with-resource-context
    (let [backend (layers/get-backend net)
          batch-size (layers/batch-size net)
          batching-system (-> (batch/create-dataset-batching-system input-labels (mapv first output-labels-and-loss) batch-size
                                                                    dataset (drv/get-driver backend) (drv/get-stream backend)
                                                                    (dtype/get-datatype backend))
                              (batch/setup true))
          loss-fns (mapv (fn [[label loss] output-size]
                           (opt/setup-loss loss backend batch-size output-size))
                         output-labels-and-loss (cp/multi-output-size net))
          optimiser (opt/setup-optimiser optimiser backend (layers/parameter-count net))
          train-config {:network net :optimiser optimiser :loss-fn loss-fns :batching-system batching-system}]
      (:network
       (reduce (fn [train-config epoch-num]
                 (-> train-config
                     (update-in [:batching-system] #(batch/setup-epoch % :training))
                     train-batches
                     (epoch-progress epoch-num)))
               train-config
               (range epoch-count))))))


(defn run
  [net dataset input-labels]
  (resource/with-resource-context
    (let [backend (layers/get-backend net)
          batch-size (layers/batch-size net)
          batching-system (-> (batch/create-dataset-batching-system input-labels [] batch-size
                                                                    dataset (drv/get-driver backend) (drv/get-stream backend)
                                                                    (dtype/get-datatype backend))
                              (batch/setup false))]
      (second (run-and-reshape {:network net :batching-system batching-system} :running)))))
