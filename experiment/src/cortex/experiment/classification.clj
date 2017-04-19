(ns cortex.experiment.classification
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.core.matrix :as m]
            [think.parallel.core :as parallel]
            [cortex.util :as util]
            [cortex.experiment.train :as experiment-train]
            [cortex.nn.network :as network]
            [cortex.nn.execute :as execute])
  (:import [java.io File]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn directory->file-label-seq
  "Given a directory with subdirs named after labels, produce an
infinite interleaved sequence of [sub-dir-name sub-dir-file]
to create balanced training classes using partition along with interleave.
Class balance is only guaranteed if the sequence is infinite or if
each directory has the same number of files."
  [dirname infinite?]
  (let [sub-dirs (.listFiles ^File (io/file dirname))
        file-sequences  (->> sub-dirs
                             (map (fn [^File sub-dir]
                                    (map vector
                                     (if infinite?
                                       (mapcat shuffle
                                               (repeatedly #(seq (.listFiles sub-dir))))
                                       (seq (.listFiles sub-dir)))
                                     (repeat (.getName sub-dir))))))]
    (if infinite?
      (apply interleave file-sequences)
      (-> (mapcat identity file-sequences)
          shuffle))))


(defn src-seq->obs-seq
  "Perform a transformation from a src sequence to a obs-sequence
assuming the src->obs transformation itself produces potentially a sequence
of observations for a single src item.  Perform this transformation
in an offline thread pool storing allowing up to queue-size transformed
sequences in memory.  Return a combination of observations and
a shutdown function to be used in the case where the input sequence
is infinite."
  [src-item-seq src-item->obs-seq-fn & {:keys [queue-size]
                                          :or {queue-size 100}}]
  (let [{:keys [sequence shutdown-fn]} (parallel/queued-sequence src-item->obs-seq-fn
                                                                 [src-item-seq]
                                                                 :queue-size queue-size)]
    {:observations (mapcat identity sequence)
     :shutdown-fn shutdown-fn}))


(defn label->vec-fn
  [class-names]
  (let [num-classes (count class-names)
        src-vec (vec (repeat num-classes 0))
        class-name->index (into {} (map-indexed (comp vec reverse list) class-names))]
    (fn [label]
      (assoc src-vec (class-name->index label) 1))))


(defn vec->label-fn
  [class-names]
  (let [index->class-name (into {} (map-indexed vector class-names))]
    (fn [label-vec]
      (get index->class-name (util/max-index label-vec)))))


(defn get-class-names-from-directory
  [dirname]
  (->> (.listFiles (io/file dirname))
       (map #(.getName ^File %))
       (sort)
       (vec)))


(defn labelled-subdirs->obs-label-seq
  "Given labelled subdirs produce a possibly infinite (balanced) sequence
of data or a finite potentially unbalanced sequence of data.
Returns map of {:observations :shutdown-fn}."
  [dirname infinite? queue-size file-label->obs-label-seq-fn]
  (-> (directory->file-label-seq dirname infinite?)
      (src-seq->obs-seq file-label->obs-label-seq-fn :queue-size queue-size)))


(defn network-eval->rich-confusion-matrix
  "A rich confusion matrix is a confusion matrix with the list of
  inferences and observations in each cell instead of just a count."
  [class-names {:keys [labels test-ds] :as network-eval}]
  (let [class-name-map (into {} (map-indexed vector class-names))
        vec->label #(class-name-map (util/max-index (vec %)))
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


(defn rich-confusion-matrix->network-confusion-matrix
  [rich-confusion-matrix observation->img-fn class-names]
  {:class-names class-names
   :matrix (mapv (fn [row-name]
                   (mapv (fn [col-name]
                           (let [{:keys [inferences observations]}
                                 (get-in rich-confusion-matrix [row-name col-name])
                                 inference-obs-pairs (->> (interleave (map m/emax inferences)
                                                                      observations)
                                                          (partition 2 )
                                                          (sort-by first >))
                                 num-pairs (count inference-obs-pairs)
                                 detailed-pairs (take 100 inference-obs-pairs)]
                             {:count num-pairs
                              :inferences (map first detailed-pairs)
                              :images (map observation->img-fn (map second detailed-pairs))}))
                         class-names))
                 class-names)})


(defn reset-confusion-matrix
  [confusion-matrix-atom observation->img-fn class-names network-eval]
  (swap! confusion-matrix-atom
         (fn [{:keys [update-index]}]
           (merge
            {:update-index (inc (long (or update-index 0)))}
            (rich-confusion-matrix->network-confusion-matrix
             (network-eval->rich-confusion-matrix class-names network-eval)
             observation->img-fn
             class-names))))
  nil)


(defn network-confusion-matrix->simple-confusion-matrix
  [{:keys [matrix] :as network-confusion-matrix}]
  (update-in network-confusion-matrix [:matrix]
             (fn [matrix] (mapv #(mapv :count %) matrix))))


(defn get-confusion-matrix
  [confusion-matrix-atom & args]
  (network-confusion-matrix->simple-confusion-matrix
   @confusion-matrix-atom))


(defn get-confusion-detail
  [confusion-matrix-atom {:keys [row col] :as params}]
  (->> (get-in @confusion-matrix-atom [:matrix row col])
       :inferences))


(defn get-confusion-image
  [confusion-matrix-atom params]
  (let [{:keys [row col index]} (->> params
                                     (map (fn [[k v]]
                                            [k (if (string? v)
                                                 (edn/read-string v)
                                                 v)]))
                                     (into {}))]
    (nth (get-in @confusion-matrix-atom [:matrix row col :images])
         index)))


(defn reset-dataset-display!
  [dataset-display-atom train-ds test-ds observation->img-fn class-names]
  (let [vec->label (vec->label-fn class-names)]
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
                                            (take 50))]
                                {:batch-type :test
                                 :images (pmap (fn [{:keys [data labels]}]
                                                 (observation->img-fn data))
                                               ds)
                                 :labels (pmap (fn [{:keys [data labels]}]
                                                 (vec->label labels))
                                               ds)})}}))
    nil))


(defn get-dataset-data
  [dataset-display-atom & args]
  (let [out (update-in @dataset-display-atom [:dataset]
                       (fn [dataset-map]
                         (map (fn [[k v]]
                                [k (dissoc v :images)])
                              dataset-map)))]
    out))


(defn get-dataset-image
  [dataset-display-atom {:keys [batch-type index]}]
  (let [img
        (nth
         (get-in @dataset-display-atom
                 [:dataset (edn/read-string batch-type) :images])
         (edn/read-string index))]
    img))


(defn routing-map
  [confusion-matrix-atom dataset-display-atom]
  {"confusion-matrix" (partial get-confusion-matrix confusion-matrix-atom)
   "confusion-detail" (partial get-confusion-detail confusion-matrix-atom)
   "confusion-image" (partial get-confusion-image confusion-matrix-atom)
   "dataset-data" (partial get-dataset-data dataset-display-atom)
   "dataset-image" (partial get-dataset-image dataset-display-atom)})


(defn test-fn
  [batch-size confusion-matrix-atom observation->img-fn class-names
   ;; TODO: no need for context here
   context new-network old-network test-ds]
  (let [labels (execute/run new-network test-ds :batch-size batch-size)
        vec->label (vec->label-fn class-names)
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
                             (double old-classification-accuracy)))]
    (reset-confusion-matrix confusion-matrix-atom
                            observation->img-fn
                            class-names
                            {:labels labels
                             :test-ds test-ds})
    (println "Classification accuracy:" classification-accuracy)
    {:best-network? best-network?
     :network (assoc new-network :classification-accuracy classification-accuracy)}))


(defn train-forever
  "Train forever. This function never returns. If an epoch count is
  provided then the best network will be loaded after N epochs and the
  training will continue from there."
  [train-ds test-ds observation->image-fn class-names initial-description
   & {:keys [epoch-count batch-size confusion-matrix-atom force-gpu?]
      :or {batch-size 128
           force-gpu? false
           confusion-matrix-atom (atom {})}}]
  (let [network (network/linear-network initial-description)]
    (experiment-train/train-n network
                              train-ds test-ds
                              :test-fn (partial test-fn
                                                batch-size
                                                confusion-matrix-atom
                                                observation->image-fn
                                                class-names)
                              :epoch-count epoch-count
                              :force-gpu? force-gpu?
                              :batch-size batch-size)))
