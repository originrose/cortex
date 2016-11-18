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
           [javax.swing JComponent JLabel JPanel BoxLayout JScrollPane SwingConstants]
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


(defn create-classification-dataset
  ([class-names data-shape cv-data-label-pairs holdout-data-label-pairs
    infinite-training-data-label-pairs-seq epoch-element-count]
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
                                             epoch-element-count)]
     (assoc dataset :class-names class-names)))
  ([class-names data-shape infinite-data-label-pair-seq epoch-element-count]
   (let [cv-count (quot (long epoch-element-count) 2)
         cv-holdout-set (take epoch-element-count infinite-data-label-pair-seq)]
     (create-classification-dataset class-names data-shape (take cv-count cv-holdout-set)
                                    (drop cv-count cv-holdout-set)
                                    (drop epoch-element-count infinite-data-label-pair-seq)
                                    epoch-element-count))))


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
  [best-network-atom network-filename ^double network-loss network]
  (let [current-best-loss (double
                           (get @best-network-atom :cv-loss Double/MAX_VALUE))]
    (when (< network-loss current-best-loss)
      (println (format "Saving network with best loss: %s" network-loss))
      (reset! best-network-atom  {:cv-loss network-loss
                                  :network-description (desc/network->description network)})
      (write-nippy-file network-filename @best-network-atom))))


(defn per-epoch-eval-training-network
  [best-network-atom network-filename epoch-idx {:keys [batching-system dataset network] :as train-config}]
  (let [eval-labels (batch/get-cpu-labels batching-system :cross-validation)
        [train-config avg-loss] (train/evaluate-training-network train-config eval-labels :cross-validation)
        avg-loss (double (first avg-loss))]
    (println (format "Loss for epoch %s: %s" epoch-idx avg-loss))
    (consider-trained-network best-network-atom network-filename avg-loss network))
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
  [dataset initial-network-description & {:keys [train-batch-size epoch-count network-filename]
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
         (train/train network (opt/adam) dataset [:data] [[:labels (opt/softmax-loss)]] epoch-count
                      :epoch-train-filter (partial per-epoch-eval-training-network best-network-atom network-filename))
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


(defn confusion-matrix-app
  [{:keys [dataset labels inferences data] :as network-eval} observation->image-fn ]
  (let [outer-grid (JPanel.)
        outer-layout (GridLayout. 0 2)
        _ (.setLayout outer-grid outer-layout)
        topmost-pane (JPanel.)
        _ (.setLayout topmost-pane (BorderLayout.))
        display-panel (JPanel.)
        display-layout (GridLayout. 5 5)
        _ (.setLayout display-panel display-layout)
        conf-matrix (network-eval->rich-confusion-matrix network-eval)
        conf-panel (JPanel.)
        num-classes (count conf-matrix)
        conf-layout (GridLayout. (+ 1 num-classes) (+ 1 num-classes))
        _ (.setLayout conf-panel conf-layout)
        class-names (get dataset :class-names)
        label-seq (vec (concat ["label"] class-names))]

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
                                 (let [inference-observation-pairs (->> (interleave (map m/emax inferences)
                                                                                    observations)
                                                                        (partition 2)
                                                                        (map vec)
                                                                        (sort-by first >)
                                                                        (take 25)
                                                                        (pmap (fn [[inference observation]]
                                                                                [inference (observation->image-fn observation)])))]
                                   (doseq [[inference observation] inference-observation-pairs]
                                     (let [icon-panel (JPanel.)
                                           icon-layout (GridLayout. 2 1)]
                                       (.setLayout icon-panel icon-layout)
                                       (.add icon-panel (JIcon. ^BufferedImage observation))
                                       (.add icon-panel (->label (str inference)))
                                       (.add display-panel icon-panel))))
                                 (catch Throwable e
                                   (clojure.pprint/pprint e)
                                   nil))
                               (.revalidate display-panel)
                               (.repaint display-panel)))
          (.add conf-panel target-label))))
    (.add outer-grid conf-panel)
    (.add outer-grid display-panel)
    (.add topmost-pane (JScrollPane. outer-grid) BorderLayout/CENTER)
    (Frames/display topmost-pane "Confusion matrix")))
