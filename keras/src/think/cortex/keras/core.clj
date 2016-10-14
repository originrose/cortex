(ns think.cortex.keras.core
  (:require [think.hdf5.core :as hdf5]
            [cortex.nn.description :as desc]
            [think.resource.core :as resource]
            [cheshire.core :as json]
            [clojure.java.io :as io]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.macros :refer [c-for]]
            [think.compute.datatype :as dtype]))


(defn read-model
  [fname]
  (json/parse-string (slurp fname) keyword))

(defmulti model-item->desc (fn [item]
                             (keyword (:class_name item))))

(defmethod model-item->desc :Convolution2D
  [{:keys [config]}]
  (let [[stride-x stride-y] (get config :subsample [1 1])
        [pad-x pad-y] (get config :padding [0 0])
        kernel-x (long (get config :nb_col))
        kernel-y (long (get config :nb_row))
        kernel-count (long (get config :nb_filter))
        id (keyword (get config :name))
        activation (keyword (get config :activation))
        conv-desc (first (desc/convolutional-expanded kernel-x kernel-y pad-x pad-y
                                                      stride-x stride-y kernel-count))
        conv-desc (assoc conv-desc :id id)]
    (when-not (= (:dim_ordering config) "tf")
      (throw (Exception. "Please convert model to 'tf' weights.  'th' weights are not supported.")))
    (if activation
      [conv-desc {:type activation}]
      [conv-desc])))


(defmethod model-item->desc :MaxPooling2D
  [{:keys [config]}]
  (let [[kernel-x kernel-y] (get config :pool_size)
        [stride-x stride-y] (get config :strides)]
    (desc/max-pooling kernel-x kernel-y 0 0 stride-x stride-y)))


(defn model->simple-description
  [model]
  (when-not (= (:class_name model) "Sequential")
    (throw (Exception. "Only sequential models supported")))
  (let [[_ width height n-channels] (get-in model [:config 0 :config :batch_input_shape])
        ;;move zeropadding into convolution modules
        model-vector (reduce (fn [model-vector {:keys [class_name config] :as current}]
                               (if (and (= (keyword class_name) :Convolution2D)
                                        (= (keyword (get (last model-vector) :class_name))
                                           :ZeroPadding2D))
                                 (conj (vec (drop-last model-vector))
                                       (update-in current [:config]
                                                  #(merge (get (last model-vector)
                                                               :config)
                                                          %)))
                                 (conj model-vector current)))
                             [] (get-in model [:config]))]
    ;;TODO models with a single channel input and figure out planar vs. interleaved
    (vec
     (flatten (concat (desc/input width height n-channels)
                      (mapv model-item->desc model-vector))))))

(defn- hdf5-child-map
  [node]
  (into {} (map (fn [node-child]
                  [(keyword (hdf5/get-name node-child))
                   node-child])
                (hdf5/get-children node))))


(defn to-core-matrix
  [data ideal-shape]
  (let [^doubles data (if (not= :double (dtype/get-datatype data))
                        (let [double-data (double-array (m/ecount data))]
                          (dtype/copy! data 0 double-data 0 (m/ecount data))
                          double-data)
                        data)]

    ;;https://github.com/mikera/core.matrix/issues/299

    ;;The simple case of using m/reshape has serious performance issues.
    (case (count ideal-shape)
      1 data
      2 (let [[n-rows n-cols] ideal-shape
              ^"[[D" retval (make-array Double/TYPE n-rows n-cols)]
          (c-for [row 0 (< row n-rows) (inc row)]
                 (dtype/copy! data (* row n-cols) (aget retval row) 0 n-cols))
          retval))))


(defmulti get-weight-shape (fn [desc weight-raw-data] (:type desc)))

(defmethod get-weight-shape :convolutional
  [desc weight-raw-data]
  [(:num-kernels desc)
   (quot (m/ecount weight-raw-data) (:num-kernels desc))])

(defmethod get-weight-shape :linear
  [desc weights-raw-data]
  [(:output-size desc)
   (quot (m/ecount weights-raw-data) (:output-size desc))])

(defn load-weights-for-description
  [desc-seq weights-fname]
  (resource/with-resource-context
    (let [weight-file (hdf5/open-file weights-fname)
          weight-entry (first (filter (fn [node]
                                        (= (hdf5/get-name node)
                                           "model_weights"))
                                      (hdf5/get-children weight-file)))
          _ (when-not weight-entry
              (throw (Exception. "Weight file does not appear to contain model_weights.")))
          node-map (hdf5-child-map weight-entry)]
      (mapv (fn [desc]
              (let [weight-node (get node-map (:id desc))]
                (if (and weight-node (seq (hdf5/get-children weight-node)))
                  (let [weight-map (hdf5-child-map weight-node)
                        ;;Is this any more robust than just assuming first child is weights
                        ;;and second child is bias?
                        weight-id (keyword (str (name (:id desc)) "_W"))
                        bias-id (keyword (str (name (:id desc)) "_b"))
                        weight-ds (get weight-map weight-id)
                        bias-ds (get weight-map bias-id)]
                    (when-not (and weight-ds bias-ds)
                      (throw (Exception. (format "Failed to find weights and bias: wanted %s, found %s"
                                                 [weight-id bias-id] (keys weight-map)))))
                    (println "loading weights/bias for" (:id desc))
                    (let [weight-raw-data (:data (hdf5/->clj weight-ds))
                          weight-shape (get-weight-shape desc weight-raw-data)]
                     (assoc desc
                            :weights (to-core-matrix weight-raw-data weight-shape)
                            :bias (to-core-matrix (:data (hdf5/->clj bias-ds))
                                                  [(second weight-shape)]))))
                  desc)))
            desc-seq))))


(defn model->description
  "Given a json model and weight hdf5 file load model into a cortex description layer."
  [model-json-fname weight-hdf5-fname]
  (-> (read-model model-json-fname)
      model->simple-description
      (load-weights-for-description weight-hdf5-fname)))



(defn reshape-time-test
  []
  (let [n-rows 100
        n-cols 1000
        src-array (double-array (* n-rows n-cols))]
    (println "reshape time")
    (time (dotimes [idx 10]
            (m/reshape src-array [n-rows n-cols])))
    (println "c-for time")
    (time (dotimes [idx 10]
            (let [^"[[D" dest (make-array Double/TYPE n-rows n-cols)]
              (c-for [row 0 (< row n-rows) (inc row)]
                     (java.lang.System/arraycopy src-array (* row n-cols) (get dest row) 0 n-cols)))))))
