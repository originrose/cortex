(ns think.cortex.keras.core
  (:require [think.hdf5.core :as hdf5]
            [cortex.nn.description :as desc]
            [think.resource.core :as resource]
            [cheshire.core :as json]
            [clojure.java.io :as io]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.macros :refer [c-for]]
            [think.datatype.core :as dtype]
            [think.compute.verify.import :as compute-verify]
            [clojure.string :as string]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* true)


(defn read-model
  "Reads a JSON keras model into a Clojure map. Just a literal representation
  with no additional munging at this point."
  [fname]
  (json/parse-string (slurp fname) keyword))


(defn match-padding
  "Maps from Keras padding descriptors to Cortex pad-x and pad-y values."
  [config]
  (cond
    (:padding config)                 (:padding config)
    (= (:border_mode config) "same")  [(mod (:nb_col config) 2)
                                       (mod (:nb_row config) 2)]
    ;; else covers "valid" padding
    :else                             [0 0]))

(defmulti model-item->desc
  "Multimethod that dispatches on keyword version of Keras model item key
  to generate the corresponding Cortex description for the item/layer."
  (fn [item]
    (keyword (:class_name item))))


(defmethod model-item->desc :Convolution2D
  [{:keys [config]}]
  (let [[stride-x stride-y] (get config :subsample [1 1])
        [pad-x pad-y] (match-padding config)
        kernel-x (long (get config :nb_col))
        kernel-y (long (get config :nb_row))
        kernel-count (long (get config :nb_filter))
        id (keyword (get config :name))
        activation (keyword (get config :activation))
        conv-desc (first (desc/convolutional-expanded kernel-x kernel-y pad-x pad-y
                                                      stride-x stride-y kernel-count))
        conv-desc (assoc conv-desc :id id)]
    (when-not (= (:dim_ordering config) "tf")
      (throw
       (Exception. "Please convert model to 'tf' weights.  'th' weights are not supported.")))
    (if (and activation
             (not= activation :linear))
      [(assoc conv-desc :embedded-activation true) {:type activation :id (keyword (str (:name config) "-activation")) :embedded id}]
      [conv-desc])))


(defmethod model-item->desc :MaxPooling2D
  [{:keys [config]}]
  (let [[kernel-x kernel-y] (:pool_size config)
        [stride-x stride-y] (:strides config)
        [layer]             (desc/max-pooling kernel-x kernel-y 0 0 stride-x stride-y)
        layer-id            (-> config :name keyword)]
    (assoc layer :id layer-id)))


(defmethod model-item->desc :Activation
  [{:keys [config]}]
  {:type (keyword (:activation config)) :id (keyword (:name config))})


(defmethod model-item->desc :Dropout
  ;; Cortex uses keep probability, Keras uses drop probability.
  [{:keys [config]}]
  (assoc (first
          (desc/dropout (- 1.0 (:p config))))
         :id (keyword (:name config))))

(defmethod model-item->desc :Flatten
  ;; Cortex doesn't require a flatten in its model description.
  [_]
  [])

(defmethod model-item->desc :Dense
  [{:keys [config]}]
  (let [output-size (long (:output_dim config))
        activation (keyword (get config :activation "linear"))
        id (keyword (:name config))
        retval (-> (first (desc/linear output-size))
                   (assoc :id id))]
    (if-not (= activation :linear)
      [(assoc retval :embedded-activation true)
       {:type activation
        :id (keyword (str (:name config) "-activation"))
        :embedded id}]
      [retval])))

(defn model->simple-description
  "Returns a simple (unbuilt) model description given the hashmap literal
  representation of a Keras JSON model description."
  [model]
  (let [model  (if (= (:class_name model) "Sequential")
                 (:config model)
                 (vec model))
        [_ width height n-channels] (get-in model [0 :config :batch_input_shape])
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
                             [] model)]
    ;;TODO models with a single channel input and figure out planar vs. interleaved
    (vec
     (flatten (concat (desc/input width height n-channels)
                      (mapv (fn [mod-item]
                              (try
                                (model-item->desc mod-item)
                                (catch Exception e
                                  (throw
                                    (ex-info "Layer not yet supported."
                                       {:cause (.getMessage ^Exception e)
                                        :layer mod-item})))))
                            model-vector))))))

(defn hdf5-child-map
  [node]
  (into {} (map (fn [node-child]
                  [(keyword (hdf5/get-name node-child))
                   node-child])
                (hdf5/get-children node))))

(defn- reshape-time-test
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
                     (java.lang.System/arraycopy src-array (* row n-cols)
                                                 (get dest row) 0 n-cols)))))))


(defn ensure-doubles
  ^doubles [data]
  (if (not= :double (dtype/get-datatype data))
    (let [double-data (double-array (m/ecount data))]
      (dtype/copy! data 0 double-data 0 (m/ecount data))
      double-data)
    data))

(defn- dims->strides
  [dims]
  (vec (reduce (fn [retval next-dim]
                 (let [last-stride (or (first retval) 1)
                       next-dim (or next-dim 1)]
                   (conj retval (* last-stride next-dim))))
               ()
               (reverse dims))))


(defn- strides-idx->dim-indexes
  [strides ^long idx]
  (let [num-strides (count strides)]
   (loop [retval []
          leftover idx
          stride-idx 0]
     (if (< stride-idx num-strides)
       (let [stride (long (strides stride-idx))
             next-item (quot leftover stride)
             next-leftover (rem leftover stride)]
         (recur (if-not (= 0 stride-idx)
                  (conj retval next-item)
                  retval) next-leftover (inc stride-idx)))
       (conj retval leftover)))))


(defn- strides-idx->dim-indexes!
  [^ints strides ^long idx ^ints retval]
  (let [num-strides (alength strides)]
   (loop [leftover idx
          stride-idx 0]
     (if (< stride-idx num-strides)
       (let [stride (aget strides stride-idx)
             next-item (quot leftover stride)
             next-leftover (rem leftover stride)]
         (when-not (= 0 stride-idx)
           (aset retval (dec stride-idx) next-item))
         (recur next-leftover (inc stride-idx)))
       (do
         (aset retval (dec stride-idx) (int leftover))
         retval)))))


(defn- strides-dim-indexes->idx
  ^long [strides dim-indexes]
  (let [n-elems (count strides)]
    (loop [retval 0
           idx 0]
      (if (< idx n-elems)
        (recur (+ retval (* (long (if (= idx (- n-elems 1))
                                    1
                                    (strides (inc idx))))
                            (long (dim-indexes idx))))
               (inc idx))
        retval))))


(defn- strides-dim-indexes-ary->idx
  ^long [^ints strides ^ints dim-indexes]
  (let [n-elems (alength strides)]
    (loop [retval 0
           idx 0]
      (if (< idx n-elems)
        (recur (+ retval (* (long (if (= idx (- n-elems 1))
                                    1
                                    (aget strides (inc idx))))
                            (long (aget dim-indexes idx))))
               (inc idx))
        retval))))


(defn- input-idx->output-idx
  ^long [input-idx input-strides reshape-indexes output-strides]
  (let [input-dim-indexes (strides-idx->dim-indexes input-strides input-idx)
        output-dim-indexes (mapv input-dim-indexes reshape-indexes)]
    (strides-dim-indexes->idx output-strides output-dim-indexes)))


(defn- input-idx->output-idx!
  [input-idx ^ints input-strides ^ints reshape-indexes ^ints output-strides ^ints input-dim-indexes ^ints output-dim-indexes]
  (let [dim-size (alength input-strides)]
    (strides-idx->dim-indexes! input-strides input-idx input-dim-indexes)
    (c-for [idx 0 (< idx dim-size) (inc idx)]
           (aset output-dim-indexes idx (aget input-dim-indexes (aget reshape-indexes idx))))
    (strides-dim-indexes-ary->idx output-strides output-dim-indexes)))


(defn- reshape-data
  "Given input with given dims and relative reshape indexes
produce a new array of double values in the order desired"
  ^doubles [data data-dims reshape-indexes]
  (let [^doubles data (ensure-doubles data)
        n-elems (long (reduce * data-dims))
        retval (double-array (alength data))
        input-strides (int-array (dims->strides data-dims))
        output-dims (int-array (mapv data-dims reshape-indexes))
        output-strides (int-array (dims->strides output-dims))
        input-dim-indexes (int-array (count input-strides))
        output-dim-indexes (int-array (count input-strides))
        reshape-indexes (int-array reshape-indexes)]
    ;;If there is a faster way of doing this I don't know it...
    (c-for [idx 0 (< idx n-elems) (inc idx)]
           (let [output-idx (input-idx->output-idx! idx input-strides reshape-indexes output-strides input-dim-indexes output-dim-indexes)]
             (aset retval output-idx (aget data idx))))
    retval))


(defn to-core-matrix
  [data ideal-shape]
  (let [^doubles data (ensure-doubles data)]
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


(defn built-desc->keras-dims
  [built-desc]
  (cond
    (= (:type built-desc) :convolutional)
    [(:kernel-height built-desc) (:kernel-width built-desc) (:input-channels built-desc) (:num-kernels built-desc)]
    (= (:type built-desc) :linear)
    (if (:input-channels built-desc)
      [(:input-width built-desc) (:input-height built-desc) (:input-channels built-desc) (:output-size built-desc)]
      [(:input-size built-desc) (:output-size built-desc)])))


(defn- load-weights
  [desc-seq weight-file]
  (let [weight-entry (first (filter (fn [node]
                                      (= (hdf5/get-name node)
                                         "model_weights"))
                                    (hdf5/get-children weight-file)))
        node-map (if weight-entry
                   (hdf5-child-map weight-entry)
                   (hdf5-child-map weight-file))]
    (mapv (fn [[desc built-desc]]
            (let [weight-node (get node-map (:id desc))]
              (if (and weight-node (seq (hdf5/get-children weight-node)))
                (let [weight-map (hdf5-child-map weight-node)
                      ;;Is this any more robust than just assuming first child is weights
                      ;;and second child is bias?
                      weight-id (keyword (str (name (:id desc)) "_W"))
                      bias-id (keyword (str (name (:id desc)) "_b"))
                      weight-ds (get weight-map weight-id)
                      bias-ds (get weight-map bias-id)
                      [weight-ds bias-ds] (if (and weight-ds bias-ds)
                                            [weight-ds bias-ds]
                                            (let [children (hdf5/get-children weight-node)]
                                              [(first children) (second children)]))]
                  (when-not (and weight-ds bias-ds)
                    (throw (Exception.
                            (format "Failed to find weights and bias: wanted %s, found %s"
                                    [weight-id bias-id] (keys weight-map)))))
                  (println "loading weights/bias for" (:id desc))
                  (let [weight-clj (hdf5/->clj weight-ds)
                        weight-raw-data (:data weight-clj)
                        weight-shape (get-weight-shape desc weight-raw-data)
                        weight-double-data (if-let [keras-dims (built-desc->keras-dims built-desc)]
                                             ;;Keras dimensions are: 0 height 1 width 2 n-channels 3 n-filters
                                             ;;We want: n-filters n-channels height width
                                             (do
                                               (println "Reshaping weights for" (:id desc))
                                               (when-not (= (apply * (:dimensions weight-clj))
                                                            (apply * keras-dims))
                                                 (throw (ex-info "Dimensions for weights and model are not compatible!"
                                                          {:cause       :wrong-dims
                                                           :cortex-dims weight-clj
                                                           :keras-dims  keras-dims})))
                                               (if (= 4 (count keras-dims))
                                                 (reshape-data weight-raw-data keras-dims [3 2 0 1])
                                                 ;;Simple transpose for linear layers as keras stores data in column major format.
                                                 (reshape-data weight-raw-data keras-dims [1 0])))
                                             (ensure-doubles weight-raw-data))]
                    (assoc desc
                           :weights (to-core-matrix weight-double-data weight-shape)
                           :bias (ensure-doubles (:data (hdf5/->clj bias-ds))))))
                desc)))
          desc-seq)))


(defn load-weights-for-description
  "Given a `desc-seq`, which consists of pairs of layers from the unbuilt and built
  versions of the model description, and the name of the hdf5 file which stores the
  weights, loads the weights for the model."
  [desc-seq weights-fname]
  (resource/with-resource-context
    (load-weights desc-seq (hdf5/open-file weights-fname))))


(defn model->description
  "Given a json model and weight hdf5 file load model into a cortex description layer."
  [model-json-fname weight-hdf5-fname]
  (let [model-desc (-> (read-model model-json-fname)
                       model->simple-description)
        built-desc (desc/build-full-network-description model-desc)
        desc-seq   (mapv vector model-desc built-desc)]
    (load-weights-for-description desc-seq weight-hdf5-fname)))

(defn tokenize-output-name
  "Parses the layer type and position per layer in the outputs file based on Keras
  naming conventions, returns as a map with `:index` and `:layer-type` keys or
  throws an exception if unable to parse."
  [out-name]
  (let [parts (string/split out-name #"_")]
    (if (= (count parts) 4)
      {:index (Long/parseLong (parts 1))
       :layer-type (keyword (parts 3))}
      (throw (ex-info "Expected format 'layertype_index' where index is an unsigned integer."
                {:cause      :bad-layer-name
                 :layer-name out-name
                 :parsed-as  parts})))))

(defn layer-output-by-id
  "Given an output file (hdf5 opened already) we return a map from keyword layer-id to
  a core matrix object that contains the values of the outputs at that layer from the
  test image during keras export process."
  [output-file]
  (let [by-id (-> output-file hdf5-child-map :layer_outputs hdf5-child-map)]
    (apply merge (for [[lyr-id hdf5-node] by-id]
                   (let [raw-data (-> hdf5-node hdf5/->clj :data)
                         as-mat   (to-core-matrix raw-data [(m/ecount raw-data)]) ]
                     {lyr-id as-mat})))))

(defn layer-output->ordered-data
  [layer-outputs]
  (->> (hdf5-child-map layer-outputs)
       (mapv (fn [[name-keywd node]]
               (let [{:keys [index layer-type] } (tokenize-output-name (name name-keywd))
                     clj-node (hdf5/->clj node)
                     node-data (:data clj-node)]
                 [index {:layer-type layer-type
                         :data (to-core-matrix node-data [(m/ecount node-data)])}])))
       (sort-by first)
       (map second)
       (remove #(or (= (:layer-type %) :Flatten)
                    (= (:layer-type %) :ZeroPadding2D)))
       vec))

(def layer-type-map
  (into {}
   (mapv vec (partition 2
                        [:Convolution2D :convolutional
                         :Activation #{:relu :softmax}
                         :MaxPooling2D :max-pooling
                         :Dense :linear
                         :Dropout :dropout]))))


(defn output-types-differ?
  [layer-types]
  (->> layer-types
       (map (fn [[cortex-type keras-type]]
              ;;Not every layer has a keras output
              (when keras-type
               (let [entry (layer-type-map keras-type)]
                 (when-not (and entry
                                (if (set? entry)
                                  (contains? entry cortex-type)
                                  (= entry cortex-type)))
                   [cortex-type keras-type])))))
       (remove nil?)
       seq))


(defn built-desc->keras-output-dims
  [built-desc]
  (when (every? #(% built-desc) [:output-channels :output-height :output-width])
    [(:output-height built-desc) (:output-width built-desc) (:output-channels built-desc)]))


(defn- associate-layer-outputs
  "Output a layer output per desc associated with that desc.
  Output may be nil for a given desc."
  [desc-seq output-map]
  (mapv (fn [[_ lyr-map]]
          (if-let [matching-output (-> lyr-map :id output-map)]
            (if-not (:embedded-activation lyr-map)
              [lyr-map matching-output]
              [lyr-map nil])
            (cond
              (:embedded lyr-map) [lyr-map (get output-map (:embedded lyr-map))]
              (= :input (:type lyr-map))     [lyr-map nil]
              :else (throw (ex-info "No matching output for layer!"
                      {:cause :missing-output
                       :layer lyr-map})))))
        desc-seq))


(defn- ordered-layer-outputs
  "Output a layer output per desc associated with that desc.
  Output may be nil for a given desc."
  [desc-seq output-seq]
  (loop [desc (first desc-seq)
         desc-seq (rest desc-seq)
         output-seq output-seq
         retval []]
    (if desc
      (let [[retval output-seq] (if (:embedded-activation desc)
                                  [(conj retval [desc nil]) output-seq]
                                  [(conj retval [desc (if (contains? desc :embedded)
                                                        (assoc (first output-seq) :layer-type :Activation)
                                                        (first output-seq))])
                                   (rest output-seq)])]
        (recur (first desc-seq) (rest desc-seq) output-seq retval))
      retval)))



(defn- check-output-dims
  "Given a mapping of vector tuples of built layer descriptions and output weights,
  as from `associate-layer-outputs`, returns information on all layers whose dims
  do not match."
  [desc-outputs]
  (->> desc-outputs
       (filter second)
       (map (fn [[lyr output]]
              (when (not= (:output-size lyr)
                          (first (m/shape output)))
                   {:id          (:id lyr)
                    :model-dims  (:output-size lyr)
                    :output-dims (m/shape output)})))
       (filter (complement nil?))))


(defn- reshape-layer-outputs
  [[built-desc {:keys [data layer-type] :as layer-output}]]
  (when layer-output
   (if-let [keras-dims (built-desc->keras-output-dims built-desc)]
     (do
       (println "Reshaping output for:" (:id built-desc) keras-dims (count data) layer-type)
       ;;keras: 0 height 1 width 2 n-channels
       ;;cortex: 0 n-channels 1 height 2 width
       (assoc layer-output
              :data (reshape-data data keras-dims [2 0 1])))
     layer-output)))


(defn load-combined-hdf5-file
  "This function is currently broken, but kept as reference for adapting a full loading
  process."
  [fname]
  (resource/with-resource-context
    (let [model-file (hdf5/open-file fname)
          file-child-map (hdf5-child-map model-file)
          printer (fn [item]
                    (clojure.pprint/pprint item)
                    item)
          src-desc (-> (:model_config file-child-map)
                       hdf5/->clj
                       :data
                       first
                       (json/parse-string keyword)
                       model->simple-description)
          ;;We build the description in order to propagate information down the network.
          ;;We then use this information to do weight and output reordering
          built-description (desc/build-full-network-description src-desc)

          weight-desc (load-weights (mapv vector
                                          src-desc
                                          built-description)
                                    model-file)
          input-desc (first weight-desc)
          input-shape (if (:output-width input-desc)
                        [(:output-channels input-desc)
                         (* (:output-height input-desc)
                            (:output-width input-desc))]
                        [(:output-size input-desc)])
          file-data (if-let [input-data (get file-child-map :test_image)]
                      (do
                        (println "Using file input data")
                        (:data (hdf5/->clj input-data)))
                      (double-array (vec (repeat (reduce * input-shape) 1.0))))
          input (to-core-matrix file-data input-shape)
          layer-outputs (->> (layer-output->ordered-data (:layer_outputs file-child-map))
                             (ordered-layer-outputs (drop 1 built-description))
                             (mapv reshape-layer-outputs))

          type-map (vec (map vector
                             (map :type (drop 1 weight-desc))
                             (map :layer-type layer-outputs)))]
      (when-let [verify-seq (seq (desc/build-and-verify-trained-network weight-desc))]
        (throw (Exception. (format "Built items failed verification:\n %s" (vec verify-seq)))))
      (when-not (= (count layer-outputs)
                   (- (count weight-desc) 1))
        (throw (Exception. (format "Layer output count mismatch: %s"
                                   type-map))))
      (when (output-types-differ? type-map)
        (throw (Exception. (format "Layer output type mismatch %s %s"
                                   type-map
                                   (vec (output-types-differ? type-map))))))
      {:model weight-desc
       :input input
       :layer-outputs (mapv :data layer-outputs)})))


(defn load-sidecar-model
  "Given a json file, weights h5 file, and output file (generated by Python
  export utils provided by cortex-keras), attempt to load model and, if
  failing, throw an ex-info that includes a report of model<->weight mismatched
  dimensions."
  [json-file weights-h5-file output-file]
  (resource/with-resource-context
    (try
      (model->description json-file weights-h5-file)
      (catch Exception e
        (throw (ex-info "Cannot create model, returning diagnostics."
                  {:cause  :model-weight-mismatch
                   :report (let [model-desc (-> json-file
                                                read-model
                                                model->simple-description)
                                 built-desc (desc/build-full-network-description model-desc)
                                 outputs    (layer-output-by-id (hdf5/open-file output-file))
                                 desc-seq   (mapv vector model-desc built-desc)
                                 by-layer   (associate-layer-outputs desc-seq outputs)]
                             (check-output-dims by-layer))}))))))


(defn reshape-layer-output
  "For outputs that aren't flat, reshape layer weights to use Cortex ordering
  instead of Keras dim ordering."
  [[built-desc data]]
  (when data
   (if-let [keras-dims (built-desc->keras-output-dims built-desc)]
     (do
       (println "Reshaping output for: " (:id built-desc) keras-dims (count data))
       ;;keras: 0 height 1 width 2 n-channels
       ;;cortex: 0 n-channels 1 height 2 width
       (reshape-data data keras-dims [2 0 1]))
     (do
       (println "No reshape required for: " (:id built-desc))
       data))))

(defn load-sidecar-with-outputs
  "Given a sidecar model (as from load-sidecar-model) verify that outputs match
  outputs generated by Keras."
  [weight-desc desc-seq output-file]
  (resource/with-resource-context
    (let [h5-file     (hdf5/open-file output-file)
          outputs     (layer-output-by-id h5-file)
          input-desc  (first weight-desc)
          input-shape (if (:output-width input-desc)
                        [(:output-channels input-desc)
                         (* (:output-height input-desc)
                            (:output-width input-desc))]
                        [(:output-size input-desc)])
          test-data   (-> h5-file hdf5-child-map :test_image hdf5/->clj :data)
          input       (to-core-matrix test-data input-shape)
          verify-seq  (desc/build-and-verify-trained-network weight-desc)
          with-output (associate-layer-outputs desc-seq outputs)
          lyr-outputs (mapv reshape-layer-output with-output)]
      {:model weight-desc
       :input input
       :layer-outputs lyr-outputs})))

(defn load-sidecar-and-verify
  "Loads a Keras model with json-file, h5 weights file, and h5 output generated
  by provided Python export scripts if it passes verification. If it does not,
  throws ex-info with a report containing layers which did not pass verification."
  [model-json-file weights-h5-file output-h5-file]
  (let [model-desc  (-> model-json-file
                        read-model
                        model->simple-description)
        built-desc  (desc/build-full-network-description model-desc)
        desc-seq    (mapv vector model-desc built-desc)
        weight-desc (load-sidecar-model model-json-file weights-h5-file output-h5-file)
        with-output (load-sidecar-with-outputs weight-desc
                                               desc-seq
                                               output-h5-file)
        verified    (compute-verify/verify-model with-output)]
    (if (empty? (:cpu verified))
      (:model with-output)
      (throw (ex-info "Model did not pass verification."
                {:cause  :incorrect-output
                 :report verified})))))
