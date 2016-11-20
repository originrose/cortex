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


(defn- run-config
  "Returns [train-config results]"
  [{:keys [network batching-system] :as train-config} batch-type]
  (let [backend (layers/get-backend network)]
    (reduce (fn [[{:keys [network] :as train-config} results] {:keys [input-buffers]}]
              (let [;;Note lack of prepare-forward; there is no prepare-calc call
                    network (cp/multi-calc network input-buffers)]
                [(assoc train-config
                        :network network)
                 (conj results (mapv #(nn-backend/to-double-array backend %)
                                     (cp/multi-output network)))]))
            [train-config []]
            (batch/get-batches batching-system batch-type false))))



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
  [train-config cpu-labels batch-type]
  (let [[train-config guesses] (run-and-reshape train-config batch-type)
        {:keys [batching-system loss-fn]} train-config]
    {:train-config train-config
     :avg-loss (mapv opt/average-loss loss-fn guesses cpu-labels)
     :inferences guesses
     :labels cpu-labels}))


(defn println-report-epoch
  [epoch-idx {:keys [batching-system dataset] :as train-config}]
  (if-let [eval-labels (batch/get-cpu-labels batching-system :cross-validation)]
    (let [{:keys [train-config avg-loss]} (evaluate-training-network train-config
                                                                     eval-labels
                                                                     :cross-validation)]
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
        batching-system (-> (batch/create-dataset-batching-system input-labels (mapv first output-labels-and-loss) batch-size
                                                                  dataset (drv/get-driver backend) (drv/get-stream backend)
                                                                  (dtype/get-datatype backend))
                            batch/setup)
        loss-fns (mapv (fn [[label loss] output-size]
                         (opt/setup-loss loss backend batch-size output-size))
                       output-labels-and-loss (cp/multi-output-size net))
        optimiser (opt/setup-optimiser optimiser backend (layers/parameter-count net))]
    {:network net :optimiser optimiser :loss-fn loss-fns :batching-system batching-system}))


(defn train
  "Epoch train filter takes an epoch-index and a train config and produces a new
  train config; providing an opportunity for side effects (e.g., printing)."
  [net optimiser dataset input-labels output-labels-and-loss epoch-count
   & {:keys [epoch-train-filter]
      :or {epoch-train-filter println-report-epoch}}]
  (resource/with-resource-context
    (let [epoch-filter (if epoch-train-filter
                         epoch-train-filter
                         (fn [idx train-cfg] train-cfg))]
      (->> (build-train-config net optimiser dataset input-labels output-labels-and-loss)
           train-epoch-seq
           (drop 1)
           (map-indexed epoch-filter)
           (take epoch-count)
           last
           :network))))


(defn train-description
  "Same as train but takes and returns a description instead of a live network.
Also takes a function that produces a network backend.  This avoids leaking leaks gpu
resources to the user."
  [net-desc backend-fn optimiser dataset input-labels output-labels-and-loss epoch-count batch-size
   & {:keys [epoch-train-filter]
      :or {epoch-train-filter println-report-epoch}}]
  (resource/with-resource-context
    (let [network (compute-desc/build-and-create-network net-desc (backend-fn) batch-size)]
      (-> (train network optimiser dataset input-labels output-labels-and-loss epoch-count)
          desc/network->description))))


(defn run
  "Run a network products a vector of output sequences, one sequence for each output of the network."
  [net dataset input-labels & {:keys [batch-type]
                               :or {batch-type :holdout}}]
  (resource/with-resource-context
    (let [backend (layers/get-backend net)
          batch-size (layers/batch-size net)
          batching-system (-> (batch/create-dataset-batching-system input-labels [] batch-size
                                                                    dataset (drv/get-driver backend) (drv/get-stream backend)
                                                                    (dtype/get-datatype backend))
                              batch/setup)]
      (second (run-and-reshape {:network net :batching-system batching-system} batch-type)))))


(defn run-description
  "Run a network from a description, producing a vector of output sequences, one sequences for each
output of the network."
  [net-desc backend-fn dataset input-labels batch-size & {:keys [batch-type]
                                                          :or {batch-type :holdout}}]
  (resource/with-resource-context
    (let [network (compute-desc/build-and-create-network net-desc (backend-fn) batch-size)]
      (run network dataset input-labels :batch-type batch-type))))
