(ns cortex.suite.train
  (:require
    [clojure.java.io :as io]
    [think.resource.core :as resource]
    [cortex.util :as util]
    [cortex.graph :as graph]
    [cortex.dataset :as ds]
    [cortex.loss :as loss]
    [cortex.optimize :as opt]
    [cortex.optimize.adam :as adam]
    [cortex.nn.execute :as execute]
    [cortex.nn.traverse :as traverse]
    [cortex.nn.network :as network]
    [cortex.compute.batching-system :as bs]))

(def default-network-filestem "trained-network")
(def trained-networks-folder "trained-networks/")

(defn load-network
  "Loads a map of {:cv-loss :network-description :initial-description} if the
initial description saved with the network matches the provided description."
  [network-filename initial-description]
  (when (.exists (io/file network-filename))
    (let [network-data (util/read-nippy-file network-filename)]
      (when (= initial-description (get network-data :initial-description))
        network-data))))


(defn save-network
  [context network network-loss initial-description network-filename]
  (let [write-data (-> (execute/save-to-network context network {})
                       (assoc :cv-loss network-loss
                              :initial-description initial-description))]
    (util/write-nippy-file network-filename write-data)
    write-data))


(defn- per-epoch-eval-training-network
  [context best-network-atom network-filename initial-description
   best-network-fn dataset simple-loss-print?
   {:keys [network inferences]}]
  (let [batch-size (:batch-size network)
        cv-columnar-input (->> (traverse/get-input-streams network)
                               (ds/get-batches dataset batch-size :cross-validation)
                               (ds/batches->columns))
        labels (->> (traverse/get-output-streams network)
                     (ds/get-batches dataset batch-size :cross-validation))
        loss-fn (execute/network->applied-loss-fn context network inferences labels)
        loss-val (apply + (map :value loss-fn))
        current-best-loss (if-let [best-loss (get @best-network-atom :cv-loss)]
                            ;; TODO: Is there a bug here? What if the best-loss isn't sequential?
                            (when (sequential? best-loss)
                              (apply + (map :value best-loss))))]
    (println (format "Loss for epoch %s: %s" (get network :epoch-count) loss-val))
    (when-not simple-loss-print?
      (println (loss/loss-fn->table-str loss-fn)))
    (when (or (nil? current-best-loss)
              (< (double loss-val) (double current-best-loss)))
      (println "Saving network")
      (reset! best-network-atom
              (save-network context network loss-fn
                            initial-description network-filename))
      (when best-network-fn
        ;;We use the same format here as the output of the evaluate network function below
        ;;so that clients can use the same network display system.  This is why we have data
        ;;in columnar formats.
        (best-network-fn {:labels (ds/batches->columnsv labels)
                          :inferences (ds/batches->columnsv inferences)
                          :data cv-columnar-input
                          :leaves (network/leaf-inference-layers network)}))))
  true)


(defn backup-trained-network
  [network-filestem]
  (let [network-filename (str network-filestem ".nippy")]
    (when (.exists (io/file network-filename))
      (let [backup-filename (->> (rest (range))
                                 (map #(format "%s%s-%s.nippy" trained-networks-folder network-filestem %))
                                 (remove #(.exists (io/file %)))
                                 (first))]
        (io/make-parents backup-filename)
        (io/copy (io/file network-filename) (io/file backup-filename))))))

(defn train-n
  "Generate an ininite sequence of networks where and the best networks to file:
  {:cv-loss best-loss-so-far
   :network-description best-network
   :initial-description initial-description}

  This system expects a dataset with online data augmentation so that it is
  effectively infinite although the cross-validation and holdout sets do not
  change. By default, the best network is saved to: `trained-network.nippy`

  Note, we have to have enough memory to store the cross-validation dataset
  in memory while training.

  When a better network is detected best-network-fn is called with a single
  argument of the form:
  {:labels     cross-validation dataset outputs in vectors by output-label order
   :inferences network inferences in vectors by output-label order
   :data       cross-validation dataset inputs in vectors by input-label order
   :leaves     network leaf nodes}

  If epoch-count is provided then we stop training after that many epochs else
  we continue to train forever."
  [dataset initial-description network
   & {:keys [batch-size epoch-count
             network-filestem best-network-fn
             optimizer
             reset-score
             force-gpu?
             simple-loss-print?]
      :or {batch-size 128
           network-filestem default-network-filestem
           optimizer (adam/adam)
           reset-score false
           force-gpu? false}}]
  (resource/with-resource-context
    (let [network-filename (str network-filestem ".nippy")
          ;;Backup the trained network if we haven't already
          network (if-let [loaded-network (load-network network-filename initial-description)]
                    (if reset-score
                      (assoc loaded-network :cv-loss {})
                      loaded-network)
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
          context (execute/compute-context :backend (when force-gpu?
                                                      :cuda))
          train-sequence (execute/train context network dataset [] []
                                        :batch-size batch-size
                                        :optimizer optimizer
                                        :infer-batch-type :cross-validation)
          epoch-processor (partial per-epoch-eval-training-network context
                                   (atom network) network-filename initial-description
                                   best-network-fn dataset simple-loss-print?)]
      (println "Training network:")
      (network/print-layer-summary (-> network
                                       traverse/auto-bind-io
                                       (traverse/add-training-traversal
                                         (ds/stream-descriptions dataset))))
      (->> (if epoch-count
             (take epoch-count train-sequence)
             train-sequence)
           (map epoch-processor)
           doall))))

(defn evaluate-network
  "Given a single-output network description and a dataset with the keys
:data and :labels produced set of inferences, answers, and the observations
used for both along with the original dataset.  This expects a network with
existing traversal bindings."
  [dataset network
   & {:keys [batch-size batch-type force-gpu?]
      :or {batch-size 128
           batch-type :holdout
           force-gpu? true}}]
  (let [input-streams (traverse/get-input-streams network)
        output-streams (traverse/get-output-streams network)
        inferences (execute/infer-columns (execute/compute-context :force-gpu? force-gpu?) network dataset
                                          [] []
                                          :batch-size batch-size
                                          :infer-batch-type batch-type)
        [data labels] (ds/batch-sequence->column-groups dataset batch-size batch-type
                                                        [input-streams output-streams])]
    {:labels labels
     :inferences inferences
     :data data
     :leaves (network/leaf-inference-layers network)}))

(defn print-trained-networks-summary
  "Prints a summary of the different networks trained so far.
  Respects an (optional) `network-filestem`."
  [& {:keys [network-filestem
             cv-loss->number
             cv-loss-display-precision
             extra-keys]
      :or {network-filestem default-network-filestem
           cv-loss->number #(apply + (vals %))
           cv-loss-display-precision 3}}]
  (let [cv-loss-format-string (format "%%.%sf" cv-loss-display-precision)]
    (->> trained-networks-folder
         io/file
         file-seq
         (filter #(let [n (.getPath %)]
                    (and (.contains n (.concat trained-networks-folder network-filestem))
                         (.endsWith n ".nippy"))))
         (map (fn [f] [f (util/read-nippy-file f)]))
         (map (fn [[f network]] (assoc network :filename (.getName f))))
         (map (fn [network] (update network :cv-loss cv-loss->number)))
         (sort-by :cv-loss)
         (map (fn [network] (update network :cv-loss #(format cv-loss-format-string %))))
         (clojure.pprint/print-table (concat [:filename :cv-loss :parameter-count] extra-keys)))))
