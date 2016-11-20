(ns cortex.suite.classification
  (:require [cortex.dataset :as ds]
            [cortex.nn.description :as desc]
            [clojure.java.io :as io]
            [think.parallel.core :as parallel]
            [taoensso.nippy :as nippy]
            [think.compute.batching-system :as batch]
            [think.compute.nn.train :as train]
            [think.compute.nn.cuda-backend :as gpu-compute]
            [think.compute.nn.description :as compute-desc]
            [think.compute.optimise :as opt]
            [think.resource.core :as resource]
            [cortex.util :as util]
            [clojure.core.matrix :as m])
  (:import [java.io File InputStream OutputStream ByteArrayOutputStream]
           [javax.swing JComponent JLabel JPanel BoxLayout JScrollPane SwingConstants
            BorderFactory]
           [java.awt Graphics2D Color GridLayout BorderLayout Component Rectangle]
           [java.awt.event ActionEvent ActionListener MouseListener MouseEvent]
           [java.awt.image BufferedImage]
           [mikera.gui Frames JIcon]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn balanced-file-label-pairs
  "Given a directory with subdirs named after labels, produce an
infinite interleaved sequence of [sub-dir-name sub-dir-file]
to create balanced training classes using partition along with interleave."
  [dirname & {:keys [infinite?]
              :or {infinite? true}}]
  (let [sub-dirs (.listFiles ^File (io/file dirname))]
    (->> sub-dirs
         (map (fn [^File sub-dir]
                (map vector
                     (if infinite?
                       (mapcat shuffle
                               (repeatedly #(seq (.listFiles sub-dir))))
                       (seq (.listFiles sub-dir)))
                     (repeat (.getName sub-dir)))))
         (apply interleave))))


(defn infinite-semi-balanced-observation-label-pairs
  "Create a balanced infinite sequence of labels and patches.
Assumptions are that the src-item-seq is balanced by class
and that the transformation from src item to a list of [label, patch]
maintaines that class balance meaning applying that either produces
roughly uniform sequences of label->patch.  The transformation from
src item -> patch will be done in a threaded pool and fed into a queue
with the result further slighly interleaved.

The final results will be pulled out of the sequence and shuffled so assuming
your epoch element count is large enough the result will still be balanced and
random regardless of this function's specific behavior for any specific src item."
  [src-item-seq item->patch-seq-fn & {:keys [queue-size]
                                      :or {queue-size 1000}}]
  (let [{:keys [sequence shutdown-fn]} (parallel/queued-sequence queue-size
                                                                 (* 2 (.availableProcessors (Runtime/getRuntime)))
                                                                 item->patch-seq-fn
                                                                 src-item-seq)]
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
  ([class-names data-shape cv-data-label-pairs holdout-data-label-pairs
    infinite-training-data-label-pairs-seq epoch-element-count
    & {:keys [epoch-repeat-count]
       :or {epoch-repeat-count 1}}]
   (let [label->vec (create-label->vec-fn class-names)
         seq-transform-fn #(map (fn [[data label]]
                                  [data (label->vec label)])
                                %)
         cv-seq (seq-transform-fn cv-data-label-pairs)
         holdout-seq (seq-transform-fn holdout-data-label-pairs)
         training-seq (seq-transform-fn infinite-training-data-label-pairs-seq)
         dataset (ds/create-infinite-dataset [[:data data-shape]
                                              [:labels (count class-names)]]
                                             cv-seq holdout-seq training-seq
                                             epoch-element-count
                                             :epoch-repeat-count epoch-repeat-count)]
     (assoc dataset :class-names class-names))))


(defn create-classification-dataset-from-labeled-image-subdirs
  "Given a directory name and a function that can transform
a single [^File file ^String sub-dir-name] into a sequence of
[observation label] pairs produce a classification dataset.
Queue size should be the number of obs-label-seqs it will take
to add up to epoch-element-count.  This is a property of
file-lable->obs-lable-seq-fn."
  [dirname data-shape file-label->obs-label-seq-fn & {:keys [queue-size epoch-element-count]
                                                      :or {queue-size 1000
                                                epoch-element-count 10000}}]
  (let [file-pairs (balanced-file-label-pairs dirname)
        observations (infinite-semi-balanced-observation-label-pairs file-pairs
                                                                     file-label->obs-label-seq-fn)
        class-names (mapv #(.getName ^File %) (.listFiles ^File (io/file dirname)))]
    (create-classification-dataset class-names data-shape observations epoch-element-count)))


(defn image-network-description
  [num-classes image-size & {:keys [channels]
                             :or {channels 3}}]
  [(desc/input image-size image-size channels)
   (desc/convolutional 1 0 1 4) ;;Color normalization
   (desc/dropout 0.9 :distribution :gaussian) ;;multiplicative dropout
   (desc/convolutional 3 0 2 64)
   (desc/relu)
   (desc/convolutional 3 0 1 64)
   (desc/relu)
   (desc/max-pooling 2 0 2)

   (desc/convolutional 3 0 1 128)
   (desc/relu)
   (desc/convolutional 3 0 1 128)
   (desc/relu)
   (desc/max-pooling 2 0 2)

   (desc/convolutional 3 0 1 128)
   (desc/relu)
   (desc/convolutional 1 0 1 128)
   (desc/relu)
   (desc/max-pooling 2 0 2)

   (desc/dropout 0.7)
   (desc/linear->relu 2048)
   (desc/dropout 0.7)                                       ;; add noise
   (desc/linear->relu 1024)
   (desc/dropout 0.7)                                       ;; add noise
   (desc/linear->softmax num-classes)])


(defn write-nippy-file
  [fname data]
  (let [^bytes byte-data (nippy/freeze data)]
    (with-open [^OutputStream stream (io/output-stream fname)]
      (.write stream byte-data))))


(defn read-nippy-file
  [fname]
  (with-open [^InputStream stream (io/input-stream fname)
              ^ByteArrayOutputStream temp-stream (ByteArrayOutputStream.)]
    (io/copy stream temp-stream)
    (nippy/thaw (.toByteArray temp-stream))))


(defn consider-trained-network
  [best-network-atom network-filename best-network-function network-loss network]
  (let [current-best-loss (double
                           (get @best-network-atom :cv-loss Double/MAX_VALUE))]
    (when (< (double network-loss) current-best-loss)
      (println (format "Saving network with best loss: %s" network-loss))
      (reset! best-network-atom  {:cv-loss network-loss
                                  :network-description (desc/network->description network)})
      (write-nippy-file network-filename @best-network-atom)
      (when best-network-function
        (best-network-function)))))


(defn per-epoch-eval-training-network
  [best-network-atom network-filename best-network-function
   epoch-idx {:keys [batching-system dataset network] :as train-config}]
  (let [eval-labels (batch/get-cpu-labels batching-system :cross-validation)
        {:keys [train-config avg-loss inferences labels]}
        (train/evaluate-training-network
         train-config eval-labels :cross-validation)
        avg-loss (double (first avg-loss))
        best-network-function
        (when best-network-function
          #(best-network-function {:inferences inferences
                                   :labels labels
                                   :batch-type :cross-validation}))]
    (println (format "Loss for epoch %s: %s" epoch-idx avg-loss))
    (consider-trained-network best-network-atom network-filename
                              best-network-function
                              avg-loss network))
  train-config)


(defn build-gpu-network
  [network-description batch-size]
  (compute-desc/build-and-create-network network-description (gpu-compute/create-backend :float) batch-size))


(defn create-train-network-sequence
  "Generate an ininite sequence of networks where we save the best networks to file.
and reset every epoch-count iterations from the best network.  The sequence is an infinite
lazy sequence of maps of the form of:
{:cv-loss best-loss-so-far
 :network-description best-network}
This system expects a dataset with online data augmentation so that it is effectively infinite
although the cv-set and holdout-set do not change.  The best network is saved to:
trained-network.nippy"
  [dataset initial-network-description & {:keys [train-batch-size epoch-count
                                                 network-filename best-network-fn]
                                          :or {train-batch-size 128
                                               epoch-count 10
                                               network-filename "trained-network.nippy"}}]
  (repeatedly
   (fn []
     (resource/with-resource-context
       (let [network-desc-loss-map (or (when (.exists (io/file network-filename))
                                         (read-nippy-file network-filename))
                                       {:network-description initial-network-description
                                        :cv-loss Double/MAX_VALUE})
             best-network-atom (atom network-desc-loss-map)
             network-description (:network-description network-desc-loss-map)
             network (build-gpu-network network-description train-batch-size)]
         (train/train network (opt/adam) dataset [:data]
                      [[:labels (opt/softmax-loss)]] epoch-count
                      :epoch-train-filter (partial per-epoch-eval-training-network
                                                   best-network-atom network-filename
                                                   best-network-fn))
         @best-network-atom)))))


(defn evaluate-network
  "Given a single-output network description and a dataset with the keys :data and :labels produce
a set of inferences, answers, and the observations used for both along with the original dataset."
  [dataset network-description & {:keys [batch-size batch-type]
                                  :or {batch-size 128
                                       batch-type :holdout}}]
  (resource/with-resource-context
    (let [eval-labels (ds/get-data-sequence-from-dataset dataset :labels batch-type batch-size)
          eval-data (ds/get-data-sequence-from-dataset dataset :data batch-type batch-size)
          network (build-gpu-network network-description batch-size)
          run-data (first (train/run network dataset [:data] :batch-type batch-type))]
      {:dataset dataset
       :labels eval-labels
       :inferences run-data
       :data eval-data})))



(defn network-eval->confusion-matrix
  "Given a network evaluation result create a confusion matrix.  Note that you can print
this with cortex.util/print-confusion-matrix.  Expects the dataset to have a key that contains
a vector of class names that are used to derive labels for the network inference and dataset :labels."
  [network-eval]
  (let [class-names (get-in network-eval [:dataset :class-names])
        vec->label #(class-names (opt/max-index %))
        retval (util/confusion-matrix class-names)
        inference-answer-pairs (partition 2 (interleave (map vec->label (:inferences network-eval))
                                                    (map vec->label (:labels network-eval))))
        retval
        (reduce (fn [conf-mat [inference answer]]
                  (util/add-prediction conf-mat inference answer))
                (util/confusion-matrix class-names)
                inference-answer-pairs)]
    (util/print-confusion-matrix retval)
    retval))



(defn network-eval->rich-confusion-matrix
  [{:keys [dataset labels inferences data] :as network-eval}]
  (let [class-names (get-in network-eval [:dataset :class-names])
        vec->label #(class-names (opt/max-index %))
        inference-answer-patch-pairs (partition 3 (interleave inferences
                                                              (map vec->label labels)
                                                              data))
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


(defn add-click-handler
  [^Component awt-component click-fn]
  (.addMouseListener awt-component
                     (reify MouseListener
                       (mouseClicked [this e]
                         (click-fn e))
                       (mousePressed [this e])
                       (mouseReleased [this e])
                       (mouseEntered [this e])
                       (mouseExited [this e]))))

(defn ->label
  ^JLabel [data]
  (JLabel. (str data) SwingConstants/CENTER))


(defn grid-layout-panel
  ^JPanel [^long grid-rows ^long grid-cols]
  (let [retval (JPanel.)]
    (.setLayout retval (GridLayout. grid-rows grid-cols))
    retval))


(defn border-layout-panel
  ^JPanel []
  (let [retval (JPanel.)]
    (.setLayout retval (BorderLayout.))
    retval))


(defn display-panel-in-scroll-view
  [^JPanel panel ^String title]
  (let [topmost-pane (JPanel.)
        _ (.setLayout topmost-pane (BorderLayout.))]
    (.add topmost-pane (JScrollPane. panel) BorderLayout/CENTER)
    (Frames/display topmost-pane title)))


(defn batch-sequence->panel
  ^JPanel [vec->label-fn observation->img-fn dataset batch-type]
  (let [batch-sequence (ds/get-batches dataset 25 batch-type [:data :labels])
        batch-data (first batch-sequence)
        interleaved-data (partition 2 (apply interleave batch-data))]
    (let [^JPanel top-panel (border-layout-panel)
          ^JPanel outer-grid (grid-layout-panel 5 5)]
      (.add top-panel (->label (name batch-type)) BorderLayout/NORTH)
      (.add top-panel outer-grid BorderLayout/CENTER)
      (.setBorder top-panel (BorderFactory/createLineBorder Color/BLACK 2))
      (doseq [[data label-vec] interleaved-data]
        (let [^JPanel item-panel (grid-layout-panel 3 1)]
          (.add item-panel (JIcon. ^BufferedImage (observation->img-fn data)))
          (.add item-panel (->label (vec->label-fn label-vec)))
          (.setBorder item-panel (BorderFactory/createEmptyBorder 4 4 4 4))
          (.add outer-grid item-panel)))
      top-panel)))


(defn view-sample-batches
  [classification-dataset observation->img-fn]
  (let [vec->label (create-vec->label-fn (:class-names classification-dataset))
        ^JPanel outer-panel (grid-layout-panel 1 2)
        batch-fn (partial batch-sequence->panel vec->label observation->img-fn
                          classification-dataset)]
    (doseq [batch-type [:cross-validation :holdout :training]]
      (.add outer-panel ^JPanel (batch-fn batch-type)))
    (display-panel-in-scroll-view outer-panel "Sample Batches")))


(defn confusion-matrix-app
  [{:keys [dataset] :as network-eval} observation->image-fn]
  (let [^JPanel outer-grid (grid-layout-panel 0 2)
        ^JPanel display-panel (grid-layout-panel 5 5)
        class-names (:class-names dataset)
        num-classes (count class-names)
        ^JPanel conf-panel (grid-layout-panel (+ 1 num-classes) (+ 1 num-classes))
        ^JPanel conv-label-panel (border-layout-panel)
        label-seq (vec (concat ["label"] class-names))
        retval
        (fn [network-eval]
          (let [conf-matrix (network-eval->rich-confusion-matrix network-eval)]
            (.removeAll conf-panel)
            (.removeAll display-panel)
            (doseq [label label-seq]
              (.add conf-panel (->label label)))
            (doseq [label class-names]
              (.add conf-panel (->label (str label)))
              (doseq [compare-label class-names]
                (let [{:keys [inferences observations]} (get-in conf-matrix [label compare-label])
                      observation-count (count observations)
                      target-label (->label (str observation-count))]
                  (add-click-handler target-label
                                     (fn [& args]
                                       (.removeAll display-panel)
                                       (try
                                         (let [inference-observation-pairs
                                               (->> (interleave (map m/emax inferences)
                                                                observations)
                                                    (partition 2)
                                                    (map vec)
                                                    (sort-by first >)
                                                    (take 25)
                                                    (pmap (fn [[inference observation]]
                                                            [inference (observation->image-fn
                                                                        observation)])))]
                                           (doseq [[inference observation] inference-observation-pairs]
                                             (let [^JPanel icon-panel (grid-layout-panel 2 1)]
                                               (.add icon-panel (JIcon. ^BufferedImage observation))
                                               (.add icon-panel (->label (format "%1.4f" inference)))
                                               (.add display-panel icon-panel))))
                                         (catch Throwable e
                                           (clojure.pprint/pprint e)
                                           nil))
                                       (.revalidate display-panel)
                                       (.repaint display-panel)))
                  (.add conf-panel target-label)
                  (.revalidate conf-panel)
                  (.repaint conf-panel)
                  (.revalidate outer-grid)
                  (.repaint outer-grid))))))]
    (.add conv-label-panel (->label "actual") BorderLayout/WEST)
    (.add conv-label-panel (->label "predicted") BorderLayout/NORTH)
    (.add conv-label-panel conf-panel BorderLayout/CENTER)
    (.add outer-grid conv-label-panel)
    (.add outer-grid display-panel)
    (retval network-eval)
    (display-panel-in-scroll-view outer-grid "Confusion matrix")
    retval))


(defn create-confusion-app-best-network-fn
  [dataset observation->image-fn
   & {:keys [batch-type batch-size]
      :or {batch-type :cross-validation
           batch-size 128}}]
  (let [data (ds/get-data-sequence-from-dataset
              dataset :data batch-type batch-size)
        confusion-atom (atom nil)]
    (fn [{:keys [inferences labels]}]
      (let [network-eval {:dataset dataset
                          :labels (first labels)
                          :inferences (first inferences)
                          :data data}]
        (if-not @confusion-atom
          (reset! confusion-atom (confusion-matrix-app network-eval
                                                       observation->image-fn))
          (@confusion-atom network-eval))))))
