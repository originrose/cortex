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
            [cortex.experiment.util :as experiment-util])
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

(def classes
  (into [] (.list (io/file "data/train"))))

(def class-mapping
  {:class-name->index (zipmap classes (range))
   :index->class-name (zipmap (range) classes)})


(defn train
  [& [batch-size]]
  (let [[train-ds test-ds] [(-> train-folder
                                (experiment-util/create-dataset-from-folder
                                  class-mapping
                                  :colorspace :rgb
                                  :normalize false
                                  :post-process-fn #(patch/patch-mean-subtract % 103.939 116.779 123.68 :bgr-reorder true))
                                (experiment-util/infinite-class-balanced-dataset))
                            (-> test-folder
                                (experiment-util/create-dataset-from-folder
                                  class-mapping
                                  :colorspace :rgb
                                  :normalize false
                                  :post-process-fn #(patch/patch-mean-subtract % 103.939 116.779 123.68 :bgr-reorder true)))]
        network (load-network "models/resnet50.nippy" :fc1000 layers-to-add)
        batch-size (or batch-size 1)]
    (train/train-n network train-ds test-ds :batch-size batch-size :epoch-count 5)))



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


(defn -main
  [& [batch-size]]
  (train (when batch-size
           (Integer/parseInt batch-size))))
