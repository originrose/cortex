(ns cortex.experiment.classification
  (:require [clojure.edn :as edn]
            [clojure.core.matrix :as m]
            [think.image.patch :as patch]
            [think.gate.core :as gate]
            [tfevent-sink.event-io :as eio]
            [cortex.nn.network :as network]
            [cortex.nn.execute :as execute]
            [cortex.experiment.train :as experiment-train]
            [cortex.util :as util]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defn- vec->label-fn
  [{:keys [index->class-name]}]
  (fn [label-vec]
    (get index->class-name (util/max-index label-vec))))

(defn- network-eval->rich-confusion-matrix
  "A rich confusion matrix is a confusion matrix with the list of
  inferences and observations in each cell instead of just a count."
  [class-mapping {:keys [labels test-ds] :as network-eval}]
  (let [class-names (sort (keys (:class-name->index class-mapping)))
        vec->label (vec->label-fn class-mapping)
        guess-answer-patch-triplets (map (fn [label {:keys [labels data]}]
                                           [(:labels label) labels data])
                                         labels
                                         test-ds)
        initial-row (zipmap class-names (repeat {:inferences []
                                                 :observations []}))
        initial-confusion-matrix (zipmap class-names (repeat initial-row))]
    (->> guess-answer-patch-triplets
         (reduce (fn [conf-mat [inference answer patch]]
                   (update-in conf-mat [(vec->label answer)
                                        (vec->label inference)]
                              (fn [{:keys [inferences observations]}]
                                {:inferences (conj inferences inference)
                                 :observations (conj observations patch)})))
                 initial-confusion-matrix))))

(defn- rich-confusion-matrix->network-confusion-matrix
  [rich-confusion-matrix observation->img-fn class-mapping]
  (let [class-names (vec (sort (keys (:class-name->index class-mapping))))]
    {:class-names class-names
     :matrix (mapv (fn [row-name]
                     (mapv (fn [col-name]
                             (let [{:keys [inferences observations]}
                                   (get-in rich-confusion-matrix [row-name col-name])
                                   inference-obs-pairs (->> (interleave (map m/emax inferences)
                                                                        observations)
                                                            (partition 2)
                                                            (sort-by first >))
                                   num-pairs (count inference-obs-pairs)
                                   detailed-pairs (take 100 inference-obs-pairs)]
                               {:count num-pairs
                                :inferences (map first detailed-pairs)
                                :images (map observation->img-fn (map second detailed-pairs))}))
                           class-names))
                   class-names)}))

(defn- reset-confusion-matrix
  [confusion-matrix-atom observation->img-fn class-mapping network-eval]
  (swap! confusion-matrix-atom
         (fn [{:keys [update-index]}]
           (merge
            {:update-index (inc (long (or update-index 0)))}
            (rich-confusion-matrix->network-confusion-matrix
             (network-eval->rich-confusion-matrix class-mapping network-eval)
             observation->img-fn
             class-mapping))))
  nil)

(defn- network-confusion-matrix->simple-confusion-matrix
  [{:keys [matrix] :as network-confusion-matrix}]
  (update-in network-confusion-matrix [:matrix]
             (fn [matrix] (mapv #(mapv :count %) matrix))))

(defn- get-confusion-matrix
  [confusion-matrix-atom & args]
  (network-confusion-matrix->simple-confusion-matrix
   @confusion-matrix-atom))

(defn- get-confusion-detail
  [confusion-matrix-atom {:keys [row col] :as params}]
  (->> (get-in @confusion-matrix-atom [:matrix row col])
       :inferences))

(defn- get-confusion-image
  [confusion-matrix-atom params]
  (let [{:keys [row col index]} (->> params
                                     (map (fn [[k v]]
                                            [k (if (string? v)
                                                 (edn/read-string v)
                                                 v)]))
                                     (into {}))]
    (nth (get-in @confusion-matrix-atom [:matrix row col :images])
         index)))

(defn- reset-dataset-display!
  [dataset-display-atom train-ds test-ds observation->img-fn class-mapping]
  (let [vec->label (vec->label-fn class-mapping)]
    (swap! dataset-display-atom
           (fn [{:keys [update-index]}]
             {:update-index (inc (long (or update-index 0)))
              :dataset {:training (let [ds (->> (cond->> train-ds
                                                  (seq? (first train-ds)) (first))
                                                (take 50))]
                                    {:batch-type :training
                                     :images (pmap (fn [{:keys [data labels]}]
                                                     (observation->img-fn data))
                                                   ds)
                                     :labels (pmap (fn [{:keys [data labels]}]
                                                     (vec->label labels))
                                                   ds)})
                        :test (let [ds (->> test-ds
                                            (shuffle)
                                            (take 50))]
                                {:batch-type :test
                                 :images (pmap (fn [{:keys [data labels]}]
                                                 (observation->img-fn data))
                                               ds)
                                 :labels (pmap (fn [{:keys [data labels]}]
                                                 (vec->label labels))
                                               ds)})}}))
    nil))

(defn- get-dataset-data
  [dataset-display-atom & args]
  (update-in @dataset-display-atom [:dataset]
             (fn [dataset-map]
               (map (fn [[k v]]
                      [k (dissoc v :images)])
                    dataset-map))))

(defn- get-dataset-image
  [dataset-display-atom {:keys [batch-type index]}]
  (nth (get-in @dataset-display-atom
               [:dataset (edn/read-string batch-type) :images])
       (edn/read-string index)))

(defn- get-accuracy-data
  [classification-accuracy-atom & args]
  @classification-accuracy-atom)

(defn- routing-map
  [confusion-matrix-atom classification-accuracy-atom dataset-display-atom]
  {"confusion-matrix" (partial get-confusion-matrix confusion-matrix-atom)
   "confusion-detail" (partial get-confusion-detail confusion-matrix-atom)
   "confusion-image" (partial get-confusion-image confusion-matrix-atom)
   "dataset-data" (partial get-dataset-data dataset-display-atom)
   "dataset-image" (partial get-dataset-image dataset-display-atom)
   "accuracy-data" (partial get-accuracy-data classification-accuracy-atom)})

(defn- test-fn
  "The `experiment` training system supports passing in a `test-fn`
  that gets called every epoch allowing the user to compare the old
  and new network and decide which is best. Here we calculate
  classification accuracy (a good metric for classification tasks) and
  use that both for reporting and comparing."
  [confusion-matrix-atom classification-accuracy-atom observation->img-fn
   class-mapping network-filename
   {:keys [batch-size context]}
   {:keys [new-network old-network test-ds]}]
  (let [labels (execute/run new-network test-ds :batch-size batch-size)
        vec->label (vec->label-fn class-mapping)
        old-classification-accuracy (:classification-accuracy old-network)
        classification-accuracy (double
                                 (/ (->> (map (fn [label observation]
                                                (= (vec->label (:labels label))
                                                   (vec->label (:labels observation))))
                                              labels
                                              test-ds)
                                         (filter identity)
                                         (count))
                                    (count test-ds)))
        best-network? (or (nil? old-classification-accuracy)
                          (> (double classification-accuracy)
                             (double old-classification-accuracy)))
        updated-network (if best-network?
                          (let [best-network
                                (assoc new-network
                                       :classification-accuracy classification-accuracy)]
                            (reset-confusion-matrix confusion-matrix-atom
                                                    observation->img-fn
                                                    class-mapping
                                                    {:labels labels
                                                     :test-ds test-ds})
                            (experiment-train/save-network best-network network-filename))
                            ;;seems dicey. if not the best-network, 
                            ;;keeps returning the old network,which will result in the training being 
                            ;;stuck in a sub-optimal state.
                          (assoc old-network :epoch-count (get new-network :epoch-count)))]
    (swap! classification-accuracy-atom conj classification-accuracy)
    (println "Classification accuracy:" classification-accuracy)

    {:best-network? best-network?
     :network updated-network}))

(defn- train-forever
  "Train forever. This function never returns."
  [initial-description train-ds test-ds
   train-args]
  (let [network (network/linear-network initial-description)]
    (apply (partial experiment-train/train-n network
                    train-ds test-ds)
           (-> train-args seq flatten))))

(defn- display-dataset-and-model
  "Starts the web server that gives real-time training updates."
  [train-ds test-ds observation->image-fn class-mapping live-updates?]
  (let [data-display-atom (atom {})
        confusion-matrix-atom (atom {})
        classification-accuracy-atom (atom [])]
    (reset-dataset-display! data-display-atom
                            train-ds
                            test-ds
                            observation->image-fn
                            class-mapping)
    (println (gate/open (atom (routing-map confusion-matrix-atom
                                           classification-accuracy-atom
                                           data-display-atom))
                        :clj-css-path "src/css"
                        :live-updates? live-updates?
                        :port 8091))
    [confusion-matrix-atom
     classification-accuracy-atom]))

(defn create-listener
  "initializes any prerequisites for listening functions, and returns a listener
  function. Arguments:
  - observation->image-fn: A function that can take observation data and return png data for web display.
    - class-mapping: A map with two entries
      - `:class-name->index` a map from class name strings to softmax indexes
      - `:index->class-name` a map from softmax indexes to class name strings
   Trains the net indefinitely on the provided training data, evaluating against the test data, and gives live updates on a local webserver hosted at http://localhost:8091. 
  "  
  [observation->image-fn class-mapping argmap]
  (fn [initial-description train-ds test-ds]
    (let [network (network/linear-network initial-description)
          network-filename (str experiment-train/default-network-filestem ".nippy")
          [confusion-matrix-atom classification-accuracy-atom]
          (display-dataset-and-model train-ds test-ds
                                     observation->image-fn
                                     class-mapping
                                     (:live-updates? argmap))]
      (partial test-fn
               confusion-matrix-atom
               classification-accuracy-atom
               observation->image-fn
               class-mapping
               network-filename))))

(defn log-weights
  "Given a network, writes all the weights from different
  layers to the tensorboard event file  "
  [network]
  (let [buf (-> network :compute-graph :buffers)]
    (mapv (fn [[k v]]
            (eio/make-event (str "buffers/" (name k))
                            (:buffer v))) buf)))

(defn tensorboard-log
  "Given the context, old network, the new network and a test dataset,
  return a map with the updated network.
  As a side-effect, stream the train/test loss as well as all buffers 
  as tensorboard events, appended to the file-path argument"
  [file-path
   {:keys [batch-size context]}
   {:keys [new-network old-network test-ds train-ds]}] ;;change per epoch
  (let [batch-size (long batch-size)
        get-label (fn [dset] (execute/run new-network dset
                                          :batch-size batch-size
                                          :loss-outputs? true
                                          :context context))
        labels (get-label test-ds)
        loss-on (fn [dset] (execute/execute-loss-fn new-network labels dset))
        cv-loss (loss-on test-ds)
        test-loss (apply + (map :value cv-loss))
        train-loss (apply + (map :value (loss-on train-ds)))
        ;;log train and test loss
        evs (mapv eio/make-event
                  (mapv (partial str "metrics/")
                        ["train-loss" "test-loss"])
                  [train-loss test-loss])
        evs (into evs (log-weights new-network))
        _ (eio/append-events file-path evs)]
    {:network (assoc new-network :cv-loss cv-loss)}))

(defn create-tensorboard-listener
  "initializes any prerequisites for listening functions, and returns a listener
  function. Takes a file-path argument where the events are logged to. "
  [{:keys [file-path]}]
  (fn [initial-description train-ds test-ds]
    (do
      (println "create-tensorboard-listener ")
      (eio/create-event-stream file-path)
      (partial tensorboard-log file-path))))

(defn perform-experiment
  "Main entry point:
    - initial-description: A cortex neural net description to train.
    - train-ds: A dataset (sequence of maps) with keys `:data`, `:labels`, used for training.
    - test-ds: A dataset (sequence of maps) with keys `:data`, `:labels`, used for testing.
    - listener: a function which takes 3 arguments: initial-description, train-ds and test-ds
              and returns a function that is executed per epoch. It could be used
              to evaluate status of training (e.g. for early stopping) or to save the network.
    - train-args: a map of optional arguments such as a force-gpu?. See cortex.experiment.train/train-n for the full list of arguments
  "
  ([initial-description train-ds test-ds listener]
   (perform-experiment initial-description train-ds test-ds listener {}))
  ([initial-description train-ds test-ds listener train-args]
   (let [test-fn (listener initial-description train-ds test-ds)]
     (train-forever initial-description train-ds test-ds
                    (assoc train-args :test-fn test-fn)))))
