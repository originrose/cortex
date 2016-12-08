(ns cortex.suite.classification
  (:require [cortex.dataset :as ds]
            [cortex.nn.description :as desc]
            [cortex.suite.io :as suite-io]
            [clojure.java.io :as io]
            [think.parallel.core :as parallel]
            [taoensso.nippy :as nippy]
            [think.compute.batching-system :as batch]
            [think.compute.nn.train :as train]
            [think.compute.nn.cuda-backend :as gpu-compute]
            [think.compute.nn.cpu-backend :as cpu-backend]
            [think.compute.nn.description :as compute-desc]
            [think.compute.optimise :as opt]
            [think.resource.core :as resource]
            [cortex.util :as util]
            [clojure.core.matrix :as m]
            [clojure.edn]
            [cortex.suite.train :as suite-train])
  (:import [java.io File InputStream OutputStream ByteArrayOutputStream]
           [javax.swing JComponent JLabel JPanel BoxLayout JScrollPane SwingConstants
            BorderFactory]
           [java.awt Graphics2D Color GridLayout BorderLayout Component Rectangle]
           [java.awt.event ActionEvent ActionListener MouseListener MouseEvent]
           [java.awt.image BufferedImage]
           [mikera.gui Frames JIcon]))


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
      (mapcat identity file-sequences))))


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


(defn create-label->vec-fn
  [class-names]
  (let [num-classes (count class-names)
        src-vec (vec (repeat num-classes 0))
        class-name->index (into {} (map-indexed (comp vec reverse list) class-names))]
    (fn [label]
      (assoc src-vec (class-name->index label) 1))))


(defn create-vec->label-fn
  [class-names]
  (let [index->class-name (into {} (map-indexed vector class-names))]
    (fn [label-vec]
      (get index->class-name (opt/max-index label-vec)))))


(defn create-classification-dataset
  ([class-names data-shape
    cv-epoch-seq
    holdout-epoch-seq
    training-epoch-seq
    & {:keys [shutdown-fn]}]
   (let [label->vec (create-label->vec-fn class-names)
         seq-transform-fn #(map (fn [[data label]]
                                  [data (label->vec label)])
                                %)
         cv-epoch-seq (map seq-transform-fn cv-epoch-seq)
         holdout-epoch-seq (if (identical? cv-epoch-seq holdout-epoch-seq)
                             cv-epoch-seq
                             (map seq-transform-fn holdout-epoch-seq))
         training-epoch-seq (map seq-transform-fn training-epoch-seq)
         dataset (ds/create-infinite-dataset [[:data data-shape]
                                              [:labels (count class-names)]]
                                             cv-epoch-seq
                                             holdout-epoch-seq
                                             training-epoch-seq
                                             :shutdown-fn
                                             shutdown-fn)]
     (assoc dataset :class-names class-names))))


(defn get-class-names-from-directory
  [dirname]
  (vec (sort
        (map #(.getName ^File %)
             (.listFiles
              (io/file dirname))))))


(defn labelled-subdirs->obs-label-seq
  "Given labelled subdirs produce a possibly infinite (balanced) sequence
of data or a finite potentially unbalanced sequence of data.
Returns map of {:observations :shutdown-fn}."
  [dirname infinite? queue-size file-label->obs-label-seq-fn]
  (-> (directory->file-label-seq dirname infinite?)
      (src-seq->obs-seq file-label->obs-label-seq-fn :queue-size queue-size)))


(defn create-classification-dataset-from-labeled-data-subdirs
  "Given a directory name and a function that can transform
a single [^File file ^String sub-dir-name] into a sequence of
[observation label] pairs produce a classification dataset.
Queue size should be the number of obs-label-seqs it will take
to add up to epoch-element-count.  This is a property of
file-lable->obs-label-seq-fn.

If your file->observation-seq function produces many identically labelled
observations per file you need to shuffle your training epochs in order to
keep your batches balanced.  This has somewhat severe performance implications
because it forces the realization of the entire training epoch of data before
the system can start training on it (as opposed to generating the epoch of data
as it is training)."
  [train-dirname test-dirname
   data-shape
   ;;training probably means augmentation
   train-file-label->obs-label-seq-fn
   ;;test means no augmentation
   test-file-label->obs-label-seq-fn
   & {:keys [queue-size epoch-element-count shuffle-training-epochs?]
      :or {queue-size 100
           epoch-element-count 10000}}]

  (let [class-names (get-class-names-from-directory test-dirname)
        _ (when-not (= class-names (get-class-names-from-directory train-dirname))
            (throw (ex-info "Class names for test and train do not match"
                            {:train-class-names (get-class-names-from-directory train-dirname)
                             :test-class-names class-names})))

        ;;I go back and forth about this but I think generally things work better if
        ;;the cross validation set is kept in memory because we will be showing
        ;;confusion matrixes and will want to map from observation->image and such
        ;;which implies the entire sequence is in memory.
        cv-epoch-seq (repeat (:observations
                              (labelled-subdirs->obs-label-seq
                               test-dirname
                               false
                               queue-size
                               test-file-label->obs-label-seq-fn)))

        holdout-epoch-seq cv-epoch-seq
        {:keys [observations shutdown-fn]} (labelled-subdirs->obs-label-seq
                                            train-dirname true queue-size
                                            train-file-label->obs-label-seq-fn)
        ;;An entire epoch of training data has to fit in memory for us to maintain that
        ;;one file can produce n identically labelled items
        training-epoch-seq (->> (partition epoch-element-count observations)
                                (map (if shuffle-training-epochs?
                                       shuffle
                                       identity)))]
    (create-classification-dataset class-names data-shape
                                   cv-epoch-seq
                                   holdout-epoch-seq
                                   training-epoch-seq
                                   :shutdown-fn
                                   shutdown-fn)))
(defonce last-network-eval (atom nil))

(defn network-eval->rich-confusion-matrix
  "A rich confusion matrix is a confusion matrix with the list of inferences and
observations in each cell instead of just a count."
  [{:keys [dataset labels inferences data] :as network-eval}]
  (reset! last-network-eval network-eval)
  (let [class-names (get-in network-eval [:dataset :class-names])
        vec->label #(class-names (opt/max-index (vec %)))
        ;;There are a lot of firsts here because generically out network could take
        ;;many inputs and produce many outputs.  When we are training classification
        ;;tasks however we know this isn't the case; we have one input and one output
        inference-answer-patch-pairs (->> (interleave (first inferences)
                                                      (map vec->label (first labels))
                                                      (first data))
                                          (partition 3))
        initial-row (zipmap class-names (repeat {:inferences []
                                                 :observations []}))
        initial-confusion-matrix (zipmap class-names (repeat initial-row))]
    (reduce (fn [conf-mat [inference answer patch]]
              (update-in conf-mat [answer (vec->label inference)]
                         (fn [{:keys [inferences observations]}]
                           {:inferences (conj inferences inference)
                            :observations (conj observations patch)})))
            initial-confusion-matrix
            inference-answer-patch-pairs)))


(defn rich-confusion-matrix->network-confusion-matrix
  [rich-confusion-matrix observation->img-fn]
  (let [class-names (vec (sort (keys rich-confusion-matrix)))]
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
                   class-names)}))


(defn reset-confusion-matrix
  [confusion-matrix-atom observation->img-fn network-eval]
  (swap! confusion-matrix-atom
         (fn [{:keys [update-index]}]
           (merge
            {:update-index (inc (long (or update-index 0)))}
            (rich-confusion-matrix->network-confusion-matrix
             (network-eval->rich-confusion-matrix network-eval)
             observation->img-fn))))
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
                                                 (clojure.edn/read-string v)
                                                 v)]))
                                     (into {}))]
    (nth (get-in @confusion-matrix-atom [:matrix row col :images])
         index)))


(defn- dataset->example-batches
  [dataset batch-definitions observation->img-fn vec->label]
  (->> (mapv (fn [{:keys [batch-type postprocess]}]
               (let [image-label-pairs
                     (->> (ds/get-batches dataset 50 batch-type [:data :labels])
                          (mapcat #(apply interleave %))
                          (partition 2)
                          postprocess
                          (take 100))]
                 [batch-type {:batch-type batch-type
                              :images (pmap (fn [[observation label]]
                                              (observation->img-fn observation))
                                            image-label-pairs)
                              :labels (map (fn [[observation label]]
                                             (vec->label label))
                                           image-label-pairs)}]))
             batch-definitions)
       (into {})))


(defn reset-dataset-display
  [dataset-display-atom dataset observation->img-fn]
  (let [vec->label (create-vec->label-fn (:class-names dataset))
        batch-defs [{:batch-type :holdout
                     :postprocess shuffle}
                    {:batch-type :cross-validation
                     :postprocess shuffle}
                    {:batch-type :training
                     :postprocess identity}]]
    (swap! dataset-display-atom
           (fn [{:keys [update-index]}]
             {:update-index (inc (long (or update-index 0)))
              :dataset (dataset->example-batches dataset batch-defs
                                                 observation->img-fn
                                                 vec->label)}))
    nil))


(defn get-dataset-data
  [dataset-display-atom & args]
  (update-in @dataset-display-atom [:dataset]
             (fn [dataset-map]
               (map (fn [[k v]]
                      [k (dissoc v :images)])
                    dataset-map))))


(defn get-dataset-image
  [dataset-display-atom {:keys [batch-type index]}]
  (let [img
        (nth
         (get-in @dataset-display-atom
                 [:dataset (clojure.edn/read-string batch-type) :images])
         (clojure.edn/read-string index))]
    img))


(defn create-routing-map
  [confusion-matrix-atom dataset-display-atom]
  {"confusion-matrix" (partial get-confusion-matrix confusion-matrix-atom)
   "confusion-detail" (partial get-confusion-detail confusion-matrix-atom)
   "confusion-image" (partial get-confusion-image confusion-matrix-atom)
   "dataset-data" (partial get-dataset-data dataset-display-atom)
   "dataset-image" (partial get-dataset-image dataset-display-atom)})


(defn best-network-function
  [confusion-matrix-atom observation->img-fn network-eval]
  (reset-confusion-matrix confusion-matrix-atom observation->img-fn network-eval))


(defn train-forever
  "Train forever.  This function never returns.  If an epoch count
is provided then the best network will be loaded after N epochs and the
training will continue from there."
  [dataset observation->image-fn initial-description
   & {:keys [epoch-count batch-size confusion-matrix-atom]
      :or {batch-size 128
           confusion-matrix-atom (atom {})}}]
  (doseq [_ (repeatedly
             #(suite-train/train-n dataset initial-description
                                   [:data] [[:labels (opt/softmax-loss)]]
                                   dataset
                                   initial-description
                                   :best-network-fn (partial best-network-function
                                                             confusion-matrix-atom
                                                             observation->image-fn)
                                   :epoch-count epoch-count
                                   :batch-size batch-size))]))
