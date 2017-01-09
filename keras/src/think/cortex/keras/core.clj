(ns think.cortex.keras.core
  (:require [think.hdf5.core :as hdf5]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
            [cortex.nn.traverse :as traverse]
            [think.resource.core :as resource]
            [cheshire.core :as json]
            [clojure.java.io :as io]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.macros :refer [c-for]]
            [think.datatype.core :as dtype]
            [cortex.verify.nn.import :as compute-verify]
            [clojure.string :as string]
            [think.compute.nn.compute-execute :as compute-execute]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* true)

(defn read-json-model
  "Reads a JSON keras model into a Clojure map. Just a literal representation
  with no additional munging at this point. Fn is public for test purposes, to
  ensure we don't lose/mismatch information from model->desc."
  [fname]
  (json/parse-string (slurp fname) keyword))


(defn match-padding
  "Maps from Keras padding descriptors to Cortex pad-x and pad-y values. Fn is
  public for test purposes."
  [config]
  (cond
    (:padding config)                 (:padding config)
    (= (:border_mode config) "same")  [(quot (:nb_col config) 2)
                                       (quot (:nb_row config) 2)]
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
        conv-desc (layers/convolutional-type-layer
                   :convolutional
                   kernel-x kernel-y pad-x pad-y stride-x stride-y kernel-count :floor)
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
        layer             (layers/convolutional-type-layer :max-pooling
                                                           kernel-x kernel-y 0 0
                                                           stride-x stride-y 0 :ceil)
        layer-id            (-> config :name keyword)]
    (assoc layer :id layer-id)))


(defmethod model-item->desc :Activation
  [{:keys [config]}]
  {:type (keyword (:activation config)) :id (keyword (:name config))})


(defmethod model-item->desc :Dropout
  ;; Cortex uses keep probability, Keras uses drop probability.
  [{:keys [config]}]
  (assoc (first
          (layers/dropout (- 1.0 (:p config))))
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
        retval (-> (first (layers/linear output-size))
                   (assoc :id id))]
    (if-not (= activation :linear)
      [(assoc retval :embedded-activation true)
       {:type activation
        :id (keyword (str (:name config) "-activation"))
        :embedded id}]
      [retval])))


(defn- keras-model->simple-description
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
     (flatten (concat (layers/input width height n-channels)
                      (mapv (fn [mod-item]
                              (try
                                (model-item->desc mod-item)
                                (catch Exception e
                                  (throw
                                    (ex-info "Layer not yet supported."
                                       {:exception e
                                        :layer mod-item})))))
                            model-vector))))))

(defn hdf5-child-map
  "For a node, return the children from that node as values corresponding to
  parent node name (as keyword) keys."
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
  (when-not (= (m/ecount data)
               (apply * data-dims))
    (throw (ex-info "Data does not match passed in dimensions"
                    {:data-size (m/ecount data)
                     :dimensions data-dims
                     :dimension-size (apply * data-dims)})))
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
  "Reshape data into ideal-shape and load into core matrix. For rationale behind
  this workaround, see: https://github.com/mikera/core.matrix/issues/299

  In brief, the simple case of using m/reshape has serious performance issues."
  [data ideal-shape]
  (let [^doubles data (ensure-doubles data)]
    (case (count ideal-shape)
      1 data
      2 (let [[n-rows n-cols] ideal-shape
              ^"[[D" retval (make-array Double/TYPE n-rows n-cols)]
          (c-for [row 0 (< row n-rows) (inc row)]
                 (dtype/copy! data (* row n-cols) (aget retval row) 0 n-cols))
          retval))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Load/reshaping of weights
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn node->keras-dims
  [node]
  (cond
    (= (:type node) :convolutional)
    [(:kernel-height node) (:kernel-width node)
     (:input-channels node) (:num-kernels node)]
    (= (:type node) :linear)
    (if (:input-channels node)
      [(:input-width node) (:input-height node)
       (:input-channels node) (:output-size node)]
      [(:input-size node) (:output-size node)])))

(defn- reshape-weights
  "check and possibly reshape weights for a given node."
  [network node-id]
  (let [node (get-in network [:layer-graph :id->node-map node-id])]
    (reduce (fn [network {:keys [key shape-fn] :as weight-desc}]
              (let [keras-dims (node->keras-dims node)
                    parameter (get node key)
                    weights (get-in network [:layer-graph :buffers
                                             (get parameter :buffer-id)
                                             :buffer])
                    weights (-> (if (= 4 (count keras-dims))
                                  (reshape-data weights keras-dims [3 2 0 1])
                                  (reshape-data weights keras-dims [1 0]))
                                (to-core-matrix (shape-fn node)))]
                (assoc-in network [:layer-graph :buffers
                                   (get parameter :buffer-id)
                                   :buffer]
                          weights)))
            network
            (->> (layers/get-parameter-descriptions node)
                 (filter #(= :weight (get % :type)))
                 seq))))

(defn- description->network
  "Given a simple list of descriptors load the weights and return a network."
  [desc-seq weight-file]
  (let [weight-entry (first (filter (fn [node]
                                      (= (hdf5/get-name node)
                                         "model_weights"))
                                    (hdf5/get-children weight-file)))
        node-map (if weight-entry
                   (hdf5-child-map weight-entry)
                   (hdf5-child-map weight-file))
        network
        (->> desc-seq
             (mapv (fn [desc]
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
                                 weight-double-data (ensure-doubles weight-raw-data)]
                             (assoc desc
                                    :weights weight-double-data
                                    :bias (ensure-doubles (:data (hdf5/->clj bias-ds))))))
                         desc))))
             network/build-network)]
    (when-let [verify-seq (seq (get network :verification-failures))]
      (throw (ex-info "Built items failed verification"
                      {:verification-failures  (vec verify-seq)})))
    (reduce reshape-weights network (keys (get-in network [:layer-graph :id->node-map])))))


(defn description-weight-file->network
  "Given a `desc-seq`, which consists of pairs of layers from the unbuilt and built
  versions of the model description, and the name of the hdf5 file which stores the
  weights, loads the weights for the model.  Returns a built network."
  [desc-seq weights-fname]
  (resource/with-resource-context
    (description->network desc-seq (hdf5/open-file weights-fname))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;Load/reshape of layer outputs
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn- outputs->output-map
  "Read the layer outputs from a file."
  [layer-outputs]
  (let [by-id (hdf5-child-map layer-outputs)]
    (apply merge (for [[lyr-id hdf5-node] by-id]
                   (let [clj-data (-> hdf5-node hdf5/->clj)
                         raw-data (get clj-data :data)
                         as-mat   (to-core-matrix raw-data [(m/ecount raw-data)])]
                     {lyr-id as-mat})))))


(defn- network->nodes
  "Given a network return a list of nodes in forward pass order"
  [network]
  (let [forward-pass (-> (traverse/auto-bind-io network)
                         traverse/network->training-traversal
                         (get-in [:traversal :forward]))]
    (->> (map :id forward-pass)
         (map #(get-in network [:layer-graph :id->node-map %])))))

(defn- associate-layer-outputs
  "Output a layer output per desc associated with that desc.
  Output may be nil for a given desc."
  [network output-map]
  ;;This function is somewhat involved because we want to do the forward traversal
  ;;in order and produce a vector of node mapped to that node's output in order
  ;;of traversal
  (->> (network->nodes network)
       (mapv (fn [node]
               (if-let [matching-output (-> node :id output-map)]
                 (if-not (:embedded-activation node)
                   [node matching-output]
                   [node nil])
                 (cond
                   (:embedded node) [node (get output-map (:embedded node))]
                   (= :input (:type node))     [node nil]
                   :else (throw (ex-info "No matching output for layer!"
                                         {:cause :missing-output
                                          :layer node
                                          :output-ids (keys output-map)}))))))))

(defn- node->keras-output-dims
  [node]
  (when (every? #(% node) [:output-channels :output-height :output-width])
    [(:output-height node) (:output-width node) (:output-channels node)]))

(defn- reshape-layer-output
  "For outputs that aren't flat, reshape layer weights to use Cortex ordering
  instead of Keras dim ordering."
  [[node data]]
  (when data
   (if-let [keras-dims (node->keras-output-dims node)]
     (do
       (println "Reshaping output for: " (:id node) keras-dims (count data))
       ;;keras: 0 height 1 width 2 n-channels
       ;;cortex: 0 n-channels 1 height 2 width
       (reshape-data data keras-dims [2 0 1]))
     (do
       (println "No reshape required for: " (:id node))
       data))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; functions below this line should be considered part of the (evolving) public
;; contract of the Keras importer.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn network-output-file->test-image
  "Given a network output h5 file, we read in the test image that has been
  stored in there."
  [output-file]
  (-> (hdf5/open-file output-file)
      hdf5-child-map
      :test_image
      hdf5/->clj
      :data))

(defn network-output-file->layer-outputs
  "Read output values from h5 file, return in hash-map of layer-id as keyword
  to value as core matrix array."
  [h5-filepath]
  (let [lyr-map (-> h5-filepath
                    hdf5/open-file
                    hdf5-child-map
                    :layer_outputs)]
    (outputs->output-map lyr-map)))

(defn keras-json->cortex-desc
  "This function fulfills one basic contract of the importer: for a given Keras
  architecture description in a JSON file with supported layer types, we map it
  to a cortex description of the same architecture.

  This also defines a separate, valid import path. I.e., if we don't want to
  import weights but we want to create a Cortex model with an equivalent arch.
  to some well-known Keras model, we can use its architecture json as a single
  argument to this function to get said description for said Cortex model."
  [model-json-fname]
  (-> model-json-fname
      read-json-model
      keras-model->simple-description))

(defn json-weight-file->network
  "This function reads the JSON architecture in Keras format, converts to a
  Cortex description, builds the Cortex description into an instantiated
  network,  then loads the Keras specified weights into the live Cortex
  network."
  [model-json-fname weight-hdf5-fname]
  (let [model-desc (keras-json->cortex-desc model-json-fname)]
    (description-weight-file->network model-desc weight-hdf5-fname)))

(defn import-model
  "Loads a Keras model with json-file, h5 weights file, and h5 output generated
  by provided Python export scripts if it passes verification. If it does not,
  throws ex-info with a report containing layers which did not pass verification.

  Note: if model fails earlier, it's the responsibility of functions that read
  Keras architecture or load h5 weights or outputs to throw close to the error.

  All import paths should go through this function. If you intend to define an
  import path (consolidated h5 file or otherwise), do so as different arity or
  dispatch through this function."
  [model-json-file weights-h5-file output-h5-file]
  (let [network     (json-weight-file->network model-json-file weights-h5-file)
        test-image  (network-output-file->test-image output-h5-file)
        output-map  (network-output-file->layer-outputs output-h5-file)
        assoc-out   (associate-layer-outputs network output-map)
        reshaped    (mapv reshape-layer-output assoc-out)
        for-verify  {:model network
                     :input test-image
                     :layer-outputs reshaped}
        verified     (compute-verify/verify-model (compute-execute/create-context) for-verify)]
    (if (empty? verified)
      network
      (throw (ex-info "Model did not pass verification."
                      {:report verified})))))
