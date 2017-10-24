(ns resnet-retrain.core
  (:require [clojure.java.io :as io]
            [clojure.string :as string]
            [mikera.image.core :as i]
            [think.image.image :as image]
            [think.image.patch :as patch]
            [think.image.data-augmentation :as image-aug]
            [cortex.nn.layers :as layers]
            [cortex.experiment.classification :as classification]
            [cortex.experiment.train :as train]
            [cortex.nn.network :as network]
            [cortex.nn.execute :as execute]
            [cortex.nn.traverse :as traverse]
            [cortex.graph :as graph]
            [cortex.util :as util]
            [cortex.experiment.util :as experiment-util]
            [cortex.compute.cpu.tensor-math :as cpu-tm]
            [cortex.tensor :as ct]
            [think.datatype.core :as dtype]
            [think.parallel.core :as parallel]
            [cortex.compute.cpu.driver :as cpu-driver])
  (:gen-class))


;; NETWORK SETUP ;;

(def layers-to-add
  [(layers/linear 21 :id :fc21)
   (layers/softmax :id :labels)])

(defn load-network
  [network-file chop-layer top-layers]
  (let [network (util/read-nippy-file network-file)
        ;; remove last layer(s)
        chopped-net (network/dissoc-layers-from-network network chop-layer)
        ;; set layers to non-trainable
        nodes (get-in chopped-net [:compute-graph :nodes]) ;;=> {:linear-1 {<params>}
        new-node-params (mapv (fn [params] (assoc params :non-trainable? true)) (vals nodes))
        frozen-nodes (zipmap (keys nodes) new-node-params)
        frozen-net (assoc-in chopped-net [:compute-graph :nodes] frozen-nodes)
        ;; add top layers
        modified-net (network/assoc-layers-to-network frozen-net (flatten top-layers))]
    modified-net))


;; DATA SETUP ;;

(defn- gather-files [path]
  (->> (io/file path)
       (file-seq)
       (filter #(.isFile %))))


(defn- load-image
  [path]
  (i/load-image path))

(def train-folder "data/train")
(def test-folder "data/test")


(defn create-train-test-folders
  "Given an original data directory that contains subdirs of classes (e.g. orig/cats, orig/dogs)
  and a split proportion, divide each class of files into a new training and testing directory
  (e.g. train/cats, train/dogs, test/cats, test/dogs)"
  [orig-data-path & {:keys [test-proportion]
                     :or {test-proportion 0.3}}]
  (let [subdirs (->> (file-seq (io/file orig-data-path))
                     (filter #(.isDirectory %) )
                     (map #(.getPath %))
                     ;; remove top (root) directory
                     rest
                     (map (juxt identity gather-files))
                     (filter #(> (count (second %)) 0)))]
    (for [[dir files] subdirs]
      (let [num-test (int (* test-proportion (count files)))
            test-files (take num-test files)
            train-files (drop num-test files)
            copy-fn (fn [file root-path]
                      (let [dest-path (str root-path "/" (last (string/split dir #"/")) "/" (.getName file))]
                        (when-not (.exists (io/file dest-path))
                          (io/make-parents dest-path)
                          (io/copy file (io/file dest-path)))))]
        (println "Working on " dir)
        (dorun (pmap (fn [file] (copy-fn file train-folder)) train-files))
        (dorun (pmap (fn [file] (copy-fn file test-folder)) test-files))
        ))))



;; TRAINING ;;

(defn classes
  []
  (into [] (.list (io/file "data/train"))))

(defn class-mapping
  []
  {:class-name->index (zipmap (classes) (range))
   :index->class-name (zipmap (range) (classes))})


(defn check-file-sizes
  []
  (->> (concat (file-seq (io/file "data/train"))
               (file-seq (io/file "data/test")))
       (filter #(.endsWith (.getName %) "png"))
       (remove #(try (let [img (i/load-image %)]
                       (and (= 224 (image/width img))
                            (= 224 (image/height img))))
                     (catch Throwable e
                       (println (format "Failed to load image %s" %))
                       (println e)
                       true)))))


(defn dataset-from-folder
  [folder-name infinite?]
  (cond-> (->> (file-seq (io/file folder-name))
               (filter #(.endsWith ^String (.getName %) "png"))
               (map (fn [file-data]
                      {:class-name (.. file-data getParentFile getName)
                       :file file-data})))
    infinite?
    (experiment-util/infinite-class-balanced-seq :class-key :class-name)))


(defn src-ds-item->net-input
  [{:keys [class-name file] :as entry}]
  (let [img-dim 224
        src-image (i/load-image file)
        ;;Ensure image is correct size
        src-image (if-not (and (= (image/width src-image) img-dim)
                               (= (image/height src-image) img-dim))
                    (i/resize src-image img-dim img-dim)
                    src-image)
        ary-data (image/->array src-image)
        ;;mask out the b-g-r channels
        mask-tensor (-> (ct/->tensor [(bit-shift-left 0xFF 16)
                                      (bit-shift-left 0xFF 8)
                                      0xFF]
                                     :datatype :int)
                        (ct/in-place-reshape [3 1 1]))
        ;;Divide to get back to range of 0-255
        div-tensor (-> (ct/->tensor [(bit-shift-left 1 16)
                                     (bit-shift-left 1 8)
                                     1]
                                    :datatype :int)
                       (ct/in-place-reshape [3 1 1]))
        ;;Use the normalization the network expects
        subtrack-tensor (-> (ct/->tensor [123.68 116.779 103.939])
                            (ct/in-place-reshape [3 1 1]))
        ;;Array of packed integer data
        img-tensor (-> (cpu-tm/as-tensor ary-data)
                       (ct/in-place-reshape [img-dim img-dim]))
        ;;Result will be b-g-r planar data
        intermediate (ct/new-tensor [3 img-dim img-dim] :datatype :int)
        result (ct/new-tensor [3 img-dim img-dim])]
    (ct/binary-op! intermediate 1.0 img-tensor 1.0 mask-tensor :bit-and)
    (ct/binary-op! intermediate 1.0 intermediate 1.0 div-tensor :/)
    (ct/assign! result intermediate)
    ;;Switch to floating point for final subtract
    (ct/binary-op! result 1.0 result 1.0 subtrack-tensor :-)
    (when-not (->> (classes)
                   (filter (partial = class-name))
                   first)
      (throw (ex-info "Class not found in classes"
                      {:classes (classes)
                       :class-name class-name})))
    {:class-name class-name
     :labels (util/one-hot-encode (classes) class-name)
     :data (cpu-tm/as-java-array result)
     :filepath (.getPath file)}))


(defn net-input->image
  [{:keys [data]}]
  (cpu-tm/tensor-context
   (let [img-dim 224
         ;;src is in normalized bgr space
         src-tens (-> (cpu-tm/as-tensor data)
                      (ct/in-place-reshape [3 img-dim img-dim]))
         subtrack-tensor (-> (ct/->tensor [123.68 116.779 103.939] :datatype (dtype/get-datatype src-tens))
                             (ct/in-place-reshape [3 1 1]))
         div-tensor (-> (ct/->tensor [(bit-shift-left 1 16)
                                      (bit-shift-left 1 8)
                                      1]
                                     :datatype (dtype/get-datatype src-tens))
                       (ct/in-place-reshape [3 1 1]))
         intermediate-float (ct/new-tensor [3 img-dim img-dim]
                                           :datatype (dtype/get-datatype src-tens))
         intermediate-int (ct/new-tensor [3 img-dim img-dim]
                                         :datatype :int)
         result (ct/new-tensor [img-dim img-dim] :datatype :int)]
     (ct/binary-op! intermediate-float 1.0 src-tens 1.0 subtrack-tensor :+)
     (ct/binary-op! intermediate-float 1.0 intermediate-float 1.0 div-tensor :*)
     (ct/assign! intermediate-int intermediate-float)
     ;;Sum together to reverse the bit shifting
     (ct/binary-op! result 1.0 result 1.0 intermediate-int :+)
     ;;Add back in alpha else we just get black images
     (ct/binary-op! result 1.0 result 1.0 (bit-shift-left 1 24) :+)
     (image/array-> (image/new-image 224 224) (cpu-tm/as-java-array result)))))


(defn convert-one-ds-item
  [ds-item]
  (src-ds-item->net-input ds-item))


(defn train-ds
  [epoch-size batch-size]
  (when-not (= 0 (rem (long epoch-size)
                      (long batch-size)))
    (throw (ex-info "Batch size is not commensurate with epoch size" {:epoch-size epoch-size
                                                                      :batch-size batch-size})))
  (ct/with-stream (cpu-driver/main-thread-cpu-stream)
   (ct/with-datatype :float
     (->> (dataset-from-folder "data/train" true)
          (take epoch-size)
          (parallel/queued-pmap (* 2 batch-size) src-ds-item->net-input)
          vec))))


(defn test-ds
  [batch-size]
  (ct/with-stream (cpu-driver/main-thread-cpu-stream)
   (ct/with-datatype :float
     (->> (dataset-from-folder "data/test" false)
          (experiment-util/batch-pad-seq batch-size)
          (parallel/queued-pmap (* 2 batch-size) src-ds-item->net-input)))))


(defn train
  [& [batch-size]]
  (let [batch-size (or batch-size 32)
        epoch-size 4096
        network (load-network "models/resnet50.nippy" :fc1000 layers-to-add)]
    (println "training using batch size of" batch-size)
    (train/train-n network
                   (partial train-ds epoch-size batch-size)
                   (partial test-ds batch-size)
                   :batch-size batch-size :epoch-count 1)))



(defn get-training-size
  [network batch-size]
  (let [traversal (traverse/training-traversal network)
        buffers (:buffers traversal)
        get-buff-size-fn (fn [buffer] (let [dims (get-in buffer [1 :dimension])]
                                        (* (:channels dims) (:height dims) (:width dims))))
        io-total (reduce + (mapv #(get-buff-size-fn %) buffers))
        param-count (graph/parameter-count (:compute-graph network))
        ;; num-vals: 4 * param-count (params, gradients, two for adam) + 2 * io-total (params, gradients)
        vals-per-batch (+ (* 4 param-count) (* 2 io-total))]
    (println "IO buffers: " io-total)
    (println "Parameter count: " param-count)
    ;; memory: 4 (4 bytes per float) * batch-size * vals-per-batch
    (* 4 batch-size vals-per-batch)))


(defn train-again
  "incrementally improve upon the trained model"
  [& [batch-size]]
  (let [batch-size (or batch-size 32)
        epoch-size 4096
        network (util/read-nippy-file "trained-network.nippy")]
    (println "training using batch size of" batch-size)
    (train/train-n network
                   (partial train-ds epoch-size batch-size)
                   (partial test-ds batch-size)
                   :batch-size batch-size :epoch-count 1)))


(defn get-class [idx]
  "A convienence function to get the class name"
    (get (classes) idx))

(defn label-one
  "Take an arbitrary test image and label it."
  []
  (let [data-item  (rand-nth (test-ds 100))]
    (->> data-item :filepath (i/load-image) (i/show))
    {:answer (->> data-item :labels util/max-index get-class)
     :guess (->> (execute/run (util/read-nippy-file "trained-network.nippy") [data-item])
                 (first)
                 (:labels)
                 (util/max-index)
                 (get-class))}))

(defn -main
  [& [batch-size continue]]
  (let [batch-size-num (when batch-size (Integer/parseInt batch-size))]
    (if continue
      (do
        (println "Training again....")
        (train-again batch-size-num))
      (do
        (println "Training fresh from RESNET-50")
        (train batch-size-num)))))
