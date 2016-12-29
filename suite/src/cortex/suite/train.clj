(ns cortex.suite.train
  (:require [cortex.suite.io :as suite-io]
            [think.compute.nn.cuda-backend :as gpu-compute]
            [cortex.nn.protocols :as cp]
            [think.resource.core :as resource]
            [clojure.java.io :as io]
            [cortex.dataset :as ds]
            [think.compute.batching-system :as bs]
            [think.compute.nn.compute-execute :as ce]
            [think.compute.nn.cuda-backend :as cuda-backend]
            [cortex.nn.execute :as execute]
            [cortex.nn.traverse :as traverse]
            [cortex.optimise :as cortex-opt]
            [cortex.nn.build :as build]))


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
  [context network network-loss initial-description network-filename]
  (let [write-data (-> (cp/save-to-network context network {})
                       (assoc :cv-loss network-loss
                              :initial-description initial-description))]
    (suite-io/write-nippy-file network-filename write-data)
    write-data))


(defn per-epoch-eval-training-network
  [context best-network-atom network-filename initial-description
   best-network-function cv-columnar-input cv-output
   {:keys [network inferences]}]
  (let [node-loss-map (->> (execute/inferences->node-id-loss-pairs network inferences cv-output)
                           (into {}))
        current-best-loss (if-let [best-loss (get @best-network-atom :cv-loss)]
                            (when (map? best-loss)
                              best-loss)
                            {})]
    (println (format "Loss for epoch %s: %s" (get network :epoch-count) node-loss-map))
    (when (every? (fn [[id ave-loss]]
                    (if-let [best (get current-best-loss id)]
                      (< ave-loss best)
                      true))
                  node-loss-map)
      (println "Saving network")
      (reset! best-network-atom
              (save-network context network node-loss-map initial-description network-filename))
      (when best-network-function
        ;;We use the same format here as the output of the evaluate network function below
        ;;so that clients can use the same network display system.  This is why we have data
        ;;in columnar formats.
        (best-network-function {:inferences (ds/batches->columnsv inferences)
                                :labels (ds/batches->columnsv cv-output)
                                :data cv-columnar-input
                                :loss node-loss-map}))))
  true)


(defn create-cuda-context
  []
  (ce/create-context #(cuda-backend/create-backend :float)))


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
  [dataset initial-description network
   & {:keys [batch-size epoch-count
             network-filestem best-network-fn
             optimiser]
      :or {batch-size 128
           network-filestem "trained-network"
           optimiser (cortex-opt/adam)}}]
  (resource/with-resource-context
    (let [network-filename (str network-filestem ".nippy")
          ;;Backup the trained network if we haven't already
          network (if-let [loaded-network (load-network network-filename
                                                        initial-description)]
                    loaded-network
                    (do
                      (backup-trained-network network-filestem)
                      (merge network
                             {:initial-description initial-description
                              :cv-loss {}})))

          input-streams (traverse/get-input-streams network)
          output-streams (traverse/get-output-streams network)
          cv-columnar-input (->> (ds/get-batches dataset batch-size :cross-validation input-streams)
                                 ds/batches->columns)
          cv-labels (ds/get-batches dataset batch-size :cross-validation output-streams)
          best-network-atom (atom network)
          context (create-cuda-context)
          train-sequence (execute/train context network dataset [] []
                                        :batch-size batch-size
                                        :optimiser optimiser
                                        :infer-batch-type :cross-validation)
          epoch-processor (partial per-epoch-eval-training-network context
                                   best-network-atom network-filename initial-description
                                   best-network-fn cv-columnar-input cv-labels)]
      (->> (if epoch-count
             (take epoch-count train-sequence)
             train-sequence)
           (map epoch-processor)
           doall))))

(defn- bindings->streams
  [bindings]
  (->> bindings
       (map :stream)
       (remove nil?)
       set))


(defn evaluate-network
  "Given a single-output network description and a dataset with the keys
:data and :labels produced set of inferences, answers, and the observations
used for both along with the original dataset.  This expects a network with
existing traversal bindings."
  [dataset network
   & {:keys [batch-size batch-type]
      :or {batch-size 128
           batch-type :holdout}}]
  (let [input-streams (traverse/get-input-streams network)
        output-streams (traverse/get-output-streams network)
        inferences (execute/infer-columns (create-cuda-context) network dataset
                                          [] []
                                          :batch-size batch-size
                                          :infer-batch-type batch-type)
        [data labels] (ds/batch-sequence->column-groups dataset batch-size batch-type
                                                        [input-streams output-streams])]
    {:labels labels
     :inferences inferences
     :data data}))
