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


(defn json-weight-file->network
  "Given a json model and weight hdf5 file load model into a cortex description layer."
  [model-json-fname weight-hdf5-fname]
  (let [model-desc (-> (read-model model-json-fname)
                       model->simple-description)]
    (description-weight-file->network model-desc weight-hdf5-fname)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;Load/reshape of layer outputs
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn- outputs->output-map
  "Given the layer outputs "
  [layer-outputs]
  (let [by-id (hdf5-child-map layer-outputs)]
    (apply merge (for [[lyr-id hdf5-node] by-id]
                   (let [raw-data (-> hdf5-node hdf5/->clj :data)
                         as-mat   (to-core-matrix raw-data [(m/ecount raw-data)]) ]
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


(defn reshape-layer-output
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


(defn- network-output-data->outputs
  "Load the layer outputs an return a list of outputs (some may be nil)
in order of the forward traversal of the gradient descent pass of the
network."
  [network hdf5-layer-outputs]
  (->> (outputs->output-map hdf5-layer-outputs)
       (associate-layer-outputs network)
       (mapv reshape-layer-output)))


(defn- network-output-file->outputs
  "Given an output file (hdf5 opened already) we return a map from keyword layer-id to
  a core matrix object that contains the values of the outputs at that layer from the
  test image during keras export process."
  [network output-file]
  (network-output-data->outputs network (-> output-file hdf5-child-map :layer_outputs)))


(defn- check-output-dims
  "Given a mapping of vector tuples of built layer descriptions and output weights,
  as from `associate-layer-outputs`, returns information on all layers whose dims
  do not match."
  [network output-vec]
  (->> (network->nodes network)
       (map (fn [output node]
              (when output
                (when-not (= (m/ecount output)
                             (:output-size node))
                  {:keras-output-size (m/ecount output)
                   :node-output-size (:output-size node)})))
            output-vec)
       (remove nil?)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;Loading of combined or separate h5 files
;;combined means weights, outputs, model-json all in one file.
;;separate means all the above are in separate files
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn network->input
  [network test-image]
  (let [input-node (first (network->nodes network))
        input-shape (if (:output-width input-node)
                      [(:output-channels input-node)
                       (* (:output-height input-node)
                          (:output-width input-node))]
                      [(:output-size input-node)])]
    (if test-image
      (:data (hdf5/->clj test-image))
      (double-array (vec (repeat (apply * input-shape) 1.0))))))


(defn load-combined-hdf5-file
  "Load an f5 file with the model json, the weights and the output all combined
  into one file."
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
          network (description->network src-desc model-file)
          input (network->input network (get file-child-map :test_image))
          outputs (network-output-data->outputs network (:layer_outputs file-child-map))]
      {:model network
       :input input
       :layer-outputs outputs})))


(defn load-sidecar-model
  "Given a json file, weights h5 file, and output file (generated by Python
  export utils provided by cortex-keras), attempt to load model and, if
  failing, throw an ex-info that includes a report of model<->weight mismatched
  dimensions."
  [json-file weights-h5-file output-file]
  (resource/with-resource-context
    (try
      (json-weight-file->network json-file weights-h5-file)
      (catch Exception e
        (throw (ex-info "Cannot create model, returning diagnostics."
                  {:cause  :model-weight-mismatch
                   :report (let [model-desc (-> json-file
                                                read-model
                                                model->simple-description)
                                 network (network/build-network model-desc)
                                 outputs (network-output-file->outputs network (hdf5/open-file output-file))]
                             (check-output-dims network outputs))}))))))


(defn- network-output-file->import-result
  "Given a sidecar model (as from load-sidecar-model) verify that outputs match
  outputs generated by Keras."
  [network output-file]
  (resource/with-resource-context
    (let [h5-file     (hdf5/open-file output-file)
          outputs     (network-output-file->outputs network h5-file)
          input       (network->input network (-> h5-file hdf5-child-map :test_image))]
      {:model network
       :input input
       :layer-outputs outputs})))


(defn load-sidecar-and-verify
  "Loads a Keras model with json-file, h5 weights file, and h5 output generated
  by provided Python export scripts if it passes verification. If it does not,
  throws ex-info with a report containing layers which did not pass verification."
  [model-json-file weights-h5-file output-h5-file]
  (let [network (load-sidecar-model model-json-file weights-h5-file output-h5-file)
        import-result (network-output-file->import-result network)
        verified    (compute-verify/verify-model (compute-execute/create-context) import-result)]
    (if (empty? verified)
      (:model import-result)
      (throw (ex-info "Model did not pass verification."
                      {:cause  :incorrect-output
                       :report verified})))))
