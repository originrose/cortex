(ns cortex.suite.train
  (:require [cortex.suite.io :as suite-io]
            [cortex.nn.description :as desc]
            [think.compute.nn.train :as train]
            [think.compute.nn.description :as compute-desc]
            [think.compute.nn.cuda-backend :as gpu-compute]
            [think.resource.core :as resource]
            [clojure.java.io :as io]
            [think.compute.optimise :as opt]
            [cortex.dataset :as ds]
            [think.compute.batching-system :as bs]))


(defn load-network
  "Loads a map of {:cv-loss :network-description :initial-description}
iff the initial description saved with the network matches the passed
in initial description.  Else returns the initial description"
  [network-filename initial-description]
  (when (.exists (io/file network-filename))
    (let [network-data (suite-io/read-nippy-file network-filename)]
      (when (= initial-description (get network-data :initial-description))
        network-data))))


(defn save-network
  [network network-loss initial-description network-filename]
  (let [write-data {:cv-loss network-loss
                    :network-description (desc/network->description network)
                    :initial-description initial-description}]
    (suite-io/write-nippy-file network-filename write-data)
    write-data))


(defn per-epoch-eval-training-network
  [best-network-atom network-filename initial-description
   best-network-function loss-compare-fn cv-input cv-output
   epoch-idx {:keys [batching-system network] :as train-config}]
  (let [{:keys [train-config avg-loss inferences]}
        (train/evaluate-training-network cv-output train-config :cross-validation)
        current-best-loss (get @best-network-atom :cv-loss)]

    (println (format "Loss for epoch %s: %s" epoch-idx avg-loss))
    (when (loss-compare-fn avg-loss current-best-loss)
      (println "Saving network")
      (reset! best-network-atom
              (save-network network avg-loss initial-description network-filename))
      (when best-network-function
        (best-network-function {:inferences inferences
                                :labels cv-output
                                :data cv-input
                                :dataset (bs/get-dataset batching-system)
                                :loss avg-loss}))))
  @best-network-atom)


(defn build-gpu-network
  [network-description batch-size]
  (compute-desc/build-and-create-network network-description
                                         (gpu-compute/create-backend :float)
                                         batch-size))


(defn backup-trained-network
  [network-filestem]
  (let [network-filename (str network-filestem ".nippy")]
    (when (.exists (io/file network-filename))
      (let [backup-filename  (->> (map (fn [idx]
                                         (str "trained-networks/" network-filestem
                                              "-" idx ".nippy"))
                                       (drop 1 (range)))
                                  (remove #(.exists (io/file %)))
                                  first)]
        (io/make-parents backup-filename)
        (io/copy (io/file network-filename) (io/file backup-filename))))))


(defn train-n
  "Generate an ininite sequence of networks where we save the best networks to file:
cv-loss is cross-validation loss.
{:cv-loss best-loss-so-far
 :network-description best-network
:initial-description initial-description}

This system expects a dataset with online data augmentation so that it is effectively infinite
although the cross-validation and holdout sets do not change.  The best network is saved to:
trained-network.nippy.
If an epoch count is provided then this function returns the best network after
the given epoch count.  When a better network is detected best-network-fn is called
with a single argument of the form:


{:data all inputs from the cross-validation dataset in vectors by input-label order
:labels all outputs from the cross-validation dataset in vectors by output-label order
:inferences all inferences the network made in vectors by output-label order
:dataset original dataset provided
:loss current best loss}


Note that this means that we have to have enough memory to store the
cross-validation dataset in memory while training.

If epoch-count is provided then we stop training after that many epochs else
we continue to train forever.
"
  [dataset initial-description input-labels output-labels-and-loss
   & {:keys [batch-size epoch-count
             network-filestem best-network-fn
             optimiser loss-compare-fn]
      :or {batch-size 128
           network-filestem "trained-network"
           optimiser (opt/adam)
           loss-compare-fn (fn [new-loss old-loss]
                             (< (first new-loss)
                                (first old-loss)))}}]
  (resource/with-resource-context
    (let [network-filename (str network-filestem ".nippy")
          ;;Backup the trained network if we haven't already
          network-desc-loss-map (if-let [loaded-network (load-network network-filename
                                                                      initial-description)]
                                  loaded-network
                                  (do
                                    (backup-trained-network network-filestem)
                                    {:network-description initial-description
                                     :initial-description initial-description
                                     :cv-loss (vec (repeat (count output-labels-and-loss)
                                                           Double/MAX_VALUE))}))
          all-labels (concat input-labels
                             (map first output-labels-and-loss))

          [cv-input cv-labels] (ds/batch-sequence->column-groups
                                dataset batch-size :cross-validation
                                [input-labels (mapv first output-labels-and-loss)])
          ;;Realize these concretely so that we don't pay for a bunch
          ;;of thunking over and over again.
          cv-input (mapv vec cv-input)
          cv-labels (mapv vec cv-labels)
          best-network-atom (atom network-desc-loss-map)
          network-description (:network-description network-desc-loss-map)
          network (build-gpu-network network-description batch-size)
          train-sequence (train/create-train-epoch-sequence network optimiser dataset
                                                            input-labels output-labels-and-loss)
          epoch-processor (partial per-epoch-eval-training-network
                                   best-network-atom network-filename initial-description
                                   best-network-fn loss-compare-fn cv-input
                                   cv-labels)]
      (->> (if epoch-count
             (take epoch-count train-sequence)
             train-sequence)
           (map-indexed epoch-processor)
           doall))))


(defn evaluate-network
  "Given a single-output network description and a dataset with the keys
:data and :labels produced set of inferences, answers, and the observations
used for both along with the original dataset."
  [dataset network-description & {:keys [batch-size batch-type input-labels output-labels]
                                  :or {batch-size 128
                                       batch-type :holdout
                                       input-labels [:data]
                                       output-labels [:labels]}}]
  (resource/with-resource-context
    (let [[cv-input cv-labels] (ds/batch-sequence->column-groups
                                dataset batch-size batch-type
                                [input-labels output-labels])
          network (build-gpu-network network-description batch-size)
          inferences (train/run network dataset input-labels :batch-type batch-type)]
      {:dataset dataset
       :labels cv-labels
       :inferences inferences
       :data cv-input})))
