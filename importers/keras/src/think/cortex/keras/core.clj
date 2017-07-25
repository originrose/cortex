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
            [cortex.nn.execute :as execute]
            [cortex.graph :as graph]
            [cortex.util :as util]
            [clojure.java.io :as io]
            [mikera.image.core :as i]
            [think.image.patch :as patch]))

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
  (let [pad (:padding config)]
    (cond
      (vector? pad)
      (cond
        ;; padding is one-level and symmetric, e.g. [2 2]
        (integer? (first pad)) pad
        ;; padding is nested but symmetric, e.g. [[3 3] [3 3]]
        (and (= (first (first pad)) (last (first pad)))
             (= (first (last pad)) (last (last pad))))
        [(first (first pad)) (first (last pad))]
        ;; padding is nested and asymmetric, e.g. [[3 2] [3 2]] => left and right not the same
        :else
        (throw (Exception. (format ("No support for asymmetric padding yet: %s") pad))))

      (= (:padding config) "same")  (mapv #(quot % 2) (:kernel_size config))

      ;; else covers "valid" padding
      :else
      [0 0])))


(defn- inbound-nodes->parents
  [keras-inbound-nodes]
  (->> (get keras-inbound-nodes 0)
       (mapv #(keyword (first %)))))


(defmulti model-item->desc
  "Multimethod that dispatches on keyword version of Keras model item key
  to generate the corresponding Cortex description for the item/layer."
  (fn [item]
    (keyword (:class_name item))))


(defmethod model-item->desc :Conv2D
  [{:keys [config inbound_nodes]}]
  (let [[stride-x stride-y] (get config :strides [1 1])
        [pad-x pad-y] (match-padding config)
        [kernel-x kernel-y] (:kernel_size config)
        kernel-count (long (:filters config))
        id (keyword (get config :name))
        activation (keyword (get config :activation))
        conv-desc (layers/convolutional-type-layer
                    :convolutional
                    kernel-x kernel-y pad-x pad-y stride-x stride-y kernel-count :floor)
        conv-desc (if inbound_nodes
                    (assoc conv-desc :id id :parents  (inbound-nodes->parents inbound_nodes))
                    (assoc conv-desc :id id))]
    (when (and (:dim_ordering config) (not= (:dim_ordering config) "tf")) ;; not used in newer Keras models
      (throw
        (Exception. "Please convert model to 'tf' weights.  'th' weights are not supported.")))
    (if (and activation
             (not= activation :linear))
      [(assoc conv-desc :embedded-activation true) {:type activation :id (keyword (str (:name config) "-activation")) :embedded id}]
      [conv-desc])))


(defmethod model-item->desc :MaxPooling2D
  [{:keys [config inbound_nodes]}]
  (let [[kernel-x kernel-y] (:pool_size config)
        [stride-x stride-y] (:strides config)
        layer             (layers/convolutional-type-layer :max-pooling
                                                           kernel-x kernel-y 0 0
                                                           stride-x stride-y 0 :floor)
        layer-id            (-> config :name keyword)]
    (if inbound_nodes
      (assoc layer :id layer-id :parents (inbound-nodes->parents inbound_nodes))
      (assoc layer :id layer-id))))


(defmethod model-item->desc :Activation
  [{:keys [config inbound_nodes]}]
  (let [layer {:type (keyword (:activation config))
               :id (keyword (:name config))}]
    (if inbound_nodes
      (assoc layer :parents (inbound-nodes->parents inbound_nodes))
      layer)))


;; Not checked with new models
(defmethod model-item->desc :Dropout
  ;; Cortex uses keep probability, Keras uses drop probability.
  [{:keys [config inbound_nodes]}]
  (let [layer (assoc (first
                       (layers/dropout (- 1.0 (:p config))))
                     :id (keyword (:name config)))]
    (if inbound_nodes
      (assoc-in layer [0 :parents] (inbound-nodes->parents inbound_nodes))
      layer)))

(defmethod model-item->desc :Flatten
  ;; Cortex doesn't require a flatten in its model description.
  [_]
  [])

(defmethod model-item->desc :Dense
  [{:keys [config inbound_nodes]}]
  (let [output-size (long (:units config))
        activation (keyword (get config :activation "linear"))
        id (keyword (:name config))
        retval (-> (first (layers/linear output-size))
                   (assoc :id id))
        retval (if inbound_nodes
                 (assoc retval :parents (inbound-nodes->parents inbound_nodes))
                 retval)]
    (if-not (= activation :linear)
      [(assoc retval :embedded-activation true)
       {:type activation
        :id (keyword (str (:name config) "-activation"))
        :embedded id}]
      [retval])))


(defmethod model-item->desc :BatchNormalization
  [{:keys [config inbound_nodes]}]
  (let [layer (layers/batch-normalization :ave-factor (:momentum config)
                                          :epsilon (:epsilon config)
                                          :id (keyword (:name config)))]
    (if inbound_nodes
      (assoc-in layer [0 :parents] (inbound-nodes->parents inbound_nodes))
      layer)))


(defmethod model-item->desc :AveragePooling2D
  [{:keys [config inbound_nodes]}]
  (let [[kernel-x kernel-y] (:pool_size config)
        [stride-x stride-y] (:strides config)
        layer             (layers/convolutional-type-layer :max-pooling
                                                           kernel-x kernel-y 0 0
                                                           stride-x stride-y 0 :ceil
                                                           :pool-op :avg)
        layer-id            (keyword (str (:name config)))]
    (if inbound_nodes
      (assoc layer :id layer-id
             :parents (inbound-nodes->parents inbound_nodes))
      (assoc layer :id layer-id))))


(defmethod model-item->desc :Add
  [{:keys [config inbound_nodes]}]
  (let [parents (inbound-nodes->parents inbound_nodes)]
    (layers/join :parents parents :operation :+
                 :id (keyword (str (:name config))))))


(defn- get-layer-by-id
  [layers id]
  (first (filter #(= (:id %) id) layers)))


(defn- keras-model->simple-description
  "Returns a simple (unbuilt) model description given the hashmap literal
  representation of a Keras JSON model description."
  [model]
  (let [model  (if (= (:class_name model) "Sequential")
                 (:config model)
                 ;; else "Model" structure, e.g. Keras pretrained applications for ResNet, VGG16m etc.
                 (get-in model [:config :layers]))
        [_ width height n-channels] (get-in model [0 :config :batch_input_shape])
        model-vector (reduce (fn [model-vector {:keys [class_name config] :as current}]
                               (cond
                                 ;;move zeropadding into convolution modules
                                 (and (= (keyword class_name) :Conv2D)
                                      (= (keyword (get (last model-vector) :class_name))
                                         :ZeroPadding2D))
                                 (conj (vec (drop-last model-vector))
                                       (assoc-in current [:config :padding]
                                                 (get-in (last model-vector) [:config :padding])))

                                 ;;drop input layer (we create our own)
                                 (= (keyword class_name) :InputLayer)
                                 model-vector

                                 ;;on "Add" layers, assoc previous layer (so skip can figure out its shortcut parent)
                                 (= (keyword class_name) :Add)
                                 (let [prev (get-in (last model-vector) [:config :name])]
                                   (conj model-vector (assoc current :previous_layer (keyword prev))))

                                 :else
                                 (conj model-vector current)))
                             [] model)
        ;;TODO models with a single channel input and figure out planar vs. interleaved
        cortex-layers (vec
                        (flatten (concat (layers/input width height n-channels)
                                         (mapv (fn [mod-item]
                                                 (try
                                                   (model-item->desc mod-item)
                                                   (catch Exception e
                                                     (throw
                                                       (ex-info (str "Layer not yet supported: " (keyword (:class_name mod-item)))
                                                                {:exception e
                                                                 :layer mod-item})))))
                                               model-vector))))]

    ;; filter through join layers and 1) add corresponding split layers 2) redirect approp. layers to point to split as parent
    (reduce (fn [cortex-desc {:keys [id type parents] :as current}]
              (if (= type :join)
                ;; split point is first common parent of the join's two branches
                (let [parent-layers (map #(get-layer-by-id cortex-desc %) parents)
                      ancestries (map (fn [parent-layer]
                                        (loop [ancestor-list [] cur parent-layer]
                                          ;; stop looking for ancestors when: reached another join layer, or reached input (end)
                                          (if (or (= (:type cur) :join)
                                                  (= (:type cur) :input))
                                            ancestor-list
                                            (recur (conj ancestor-list cur)
                                                   (get-layer-by-id cortex-desc (get (:parents cur) 0)))))) parent-layers)
                      split-parent (first (remove nil? (for [a (first ancestries)
                                                             b (last ancestries)]
                                                         (if (= a b) a nil))))]
                  (when (> (count ancestries) 2)
                    (throw (Exception.
                             (format ("Cannot support joins of more than two parents. Parents: %s") parents))))

                  ;; insert split layer and redirect :parents for its new children
                  (let [parent-id (:id split-parent)
                        split-layer (layers/split :parents [parent-id]
                                                  :id (keyword (str (name parent-id) "-split")))
                        split-layer (get split-layer 0)
                        split-children (filter (fn [layer] (some #(= parent-id %) (:parents layer))) cortex-desc)
                        split-children-idx (map #(.indexOf ^java.util.List cortex-desc %) split-children)
                        join-layer (if (some #(= parent-id %) (:parents current))
                                     (update current :parents (fn [old-parents]
                                                                (conj (remove #(= parent-id %) old-parents)
                                                                      (:id split-layer))))
                                     current)]
                    (conj (reduce (fn [desc split-child-idx]
                                    (assoc-in desc [split-child-idx :parents] [(:id split-layer)]))
                                  cortex-desc split-children-idx)
                          split-layer join-layer)))
                ;; ensure inbound_node parent actually exists in Cortex (e.g. not flatten or zero-padding)
                (if (some #(= (get (:parents current) 0) (:id %)) cortex-desc)
                  (conj cortex-desc current)
                  (conj cortex-desc (dissoc current :parents)))))
            [] cortex-layers)))


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
           (let [output-idx (input-idx->output-idx! idx input-strides reshape-indexes
                                                    output-strides input-dim-indexes output-dim-indexes)]
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
     (get-in node [:input-dimensions 0 :channels]) (:num-kernels node)]
    (= (:type node) :linear)
    (if (> (get-in (graph/node->input-dimensions node) [0 :channels]) 1)
      (let [{:keys [channels width height]} (first (graph/node->input-dimensions node))
            output-size (graph/node->output-size node)]
        [width height channels output-size])
      [(graph/node->input-size node)
       (graph/node->output-size node)])))

(defn- reshape-weights
  "check and possibly reshape weights for a given node."
  [id->weight-map network node-id]
  (let [node (-> network
                 network/network->graph
                 (graph/get-node node-id))
        weight-node (get id->weight-map (:id node))]
    ;; if node has parameters (e.g. conv, dense, batch-norm, as opposed to max-pooling)
    (if (and weight-node (seq (hdf5/get-children weight-node)))
      (let [weight-map (hdf5/child-map ((:id node) (hdf5/child-map weight-node)))]
        (if (contains? weight-map :kernel:0)
          ;; conv/dense layer?
          (let  [weight-ds (get weight-map :kernel:0)
                 bias-ds (get weight-map :bias:0)
                 [weight-ds bias-ds] (if (and weight-ds bias-ds)
                                       [weight-ds bias-ds]
                                       (let [children (hdf5/get-children weight-node)]
                                         [(first children) (second children)]))]
            (when-not (and weight-ds bias-ds)
              (throw (Exception.
                       (format "Failed to find weights and bias: wanted %s, found %s"
                               [:kernel:0 :bias:0] (keys weight-map)))))
            (let [weight-clj (hdf5/->clj weight-ds)
                  weight-raw-data (:data weight-clj)
                  weight-double-data (ensure-doubles weight-raw-data)
                  keras-dims (node->keras-dims node)
                  graph (network/network->graph network)
                  weights-arg (graph/get-node-argument node :weights)
                  bias-arg (graph/get-node-argument node :bias)
                  weights (-> (if (= 4 (count keras-dims))
                                (reshape-data weight-double-data keras-dims [3 2 0 1])
                                (reshape-data weight-double-data keras-dims [1 0]))
                              (to-core-matrix (graph/get-argument-shape graph node weights-arg)))]
              (-> network
                  (assoc-in [:compute-graph :buffers
                             (get weights-arg :buffer-id)
                             :buffer]
                            weights)
                  (assoc-in [:compute-graph :buffers
                             (get bias-arg :buffer-id)
                             :buffer]
                            (ensure-doubles (:data (hdf5/->clj bias-ds)))))))
          ;; else, is batch-norm layer
          (let [bias-ds (get weight-map :beta:0)
                scale-ds (get weight-map :gamma:0)
                mean-ds (get weight-map :moving_mean:0)
                variance-ds (get weight-map :moving_variance:0)]
            (when-not (and bias-ds scale-ds mean-ds variance-ds)
              (throw (Exception.
                       (format "Failed to find batch-norm params: wanted %s, found %s"
                               [:beta:0 :gamma:0 :moving_mean:0 :moving_variance:0] (keys weight-map)))))
            (let [params (mapv #(hdf5/->clj %) [bias-ds scale-ds mean-ds variance-ds])
                  double-params (mapv #(ensure-doubles (:data %)) params) ;; vector of 4 param vectors: [[<offset params>] [<gamma params>] ...]
                  ;; temp hack
                  channel-height (get-in node [:input-dimensions 0 :height])
                  channel-width (get-in node [:input-dimensions 0 :width])
                  expanded-params (mapv (fn [param-vec]
                                          (->> param-vec
                                               (mapcat #(repeat (* channel-height channel-width) %))))
                                        double-params)
                  [bias-arg scale-arg means-arg variances-arg] (mapv #(graph/get-node-argument node %)
                                                                     [:bias :scale :means :variances])]

              (reduce (fn [network param-kv]
                        (assoc-in network [:compute-graph :buffers
                                           (get (key param-kv) :buffer-id)
                                           :buffer]
                                  (get expanded-params (val param-kv))))
                      network (zipmap [bias-arg scale-arg means-arg variances-arg] [0 1 2 3]))
              ))))
      network)))


(defn- description->network
  "Given a simple list of descriptors load the weights and return a network."
  [desc-seq weight-file]
  (let [weight-entry (first (filter (fn [node]
                                      (= (hdf5/get-name node)
                                         "model_weights"))
                                    (hdf5/get-children weight-file)))
        id->weight-map (if weight-entry
                         (hdf5/child-map weight-entry)
                         (hdf5/child-map weight-file))
        network (network/linear-network desc-seq)
        network (reduce (partial reshape-weights id->weight-map)
                        network
                        (graph/dfs-seq (network/network->graph network)))]
    ;;Generate parameters and check that all our shapes are correct.
    (update network :compute-graph graph/generate-parameters)))


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
  (let [by-id (hdf5/child-map layer-outputs)]
    (apply merge (for [[lyr-id hdf5-node] by-id]
                   (let [clj-data (-> hdf5-node hdf5/->clj)
                         raw-data (get clj-data :data)
                         as-mat   (to-core-matrix raw-data [(m/ecount raw-data)])]
                     {lyr-id as-mat})))))


(defn- network->nodes
  "Given a network return a list of nodes in forward pass order"
  [network]
  (let [forward-pass (-> (traverse/training-traversal network)
                         :forward)]
    (->> (map :id forward-pass)
         (map #(get-in network [:compute-graph :nodes %])))))

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
                   (= :split (:type node))     [node nil]
                   :else (throw (ex-info "No matching output for layer!"
                                         {:cause :missing-output
                                          :layer node
                                          :output-ids (keys output-map)}))))))))

(defn- node->keras-output-dims
  [node]
  (let [{:keys [channels width height]} (first (graph/node->output-dimensions node))]
    (when (or (> channels 1)
              (> height 1))
     [height width channels])))


(defn- reshape-layer-output
  "For outputs that aren't flat, reshape layer weights to use Cortex ordering
  instead of Keras dim ordering."
  [[{:keys [id] :as node} data]]
  (when data
    [id
     (if-let [keras-dims (node->keras-output-dims node)]
       (do
         (println "Reshaping output for: " id keras-dims (count data))
         ;;keras: 0 height 1 width 2 n-channels
         ;;cortex: 0 n-channels 1 height 2 width
         (reshape-data data keras-dims [2 0 1]))
       (do
         (println "No reshape required for:" id)
         data))]))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; functions below this line should be considered part of the (evolving) public
;; contract of the Keras importer.
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn network-output-file->test-image
  "Given a network output h5 file, we read in the test image that has been
  stored in there."
  [output-file]
  (-> (hdf5/open-file output-file)
      hdf5/child-map
      :test_image
      hdf5/->clj
      :data))

(defn network-output-file->layer-outputs
  "Read output values from h5 file, return in hash-map of layer-id as keyword
  to value as core matrix array."
  [h5-filepath]
  (let [lyr-map (-> h5-filepath
                    hdf5/open-file
                    hdf5/child-map
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
        reshaped    (->> (mapv reshape-layer-output assoc-out)
                         (remove nil?)
                         (into {}))
        roots  (graph/roots (network/network->graph network))
        for-verify  {:model network
                     :layer-id->output (assoc reshaped
                                              (first roots)
                                              test-image)}
        verified     (compute-verify/verify-model (execute/compute-context) for-verify)]
    (if (empty? verified)
      network
      (throw (ex-info "Model did not pass verification."
                      {:report verified})))))

(defn import-and-save
  "Once import-model is verified to work, this function will save the imported model to a nippy file."
  [model-json-file weights-h5-file trained-network-name]
  (util/write-nippy-file trained-network-name
                         (json-weight-file->network model-json-file weights-h5-file)))


;; ================== Testing images ====================== ;;

(defn label-one
  "Take a random test image and label it."
  ([image-filename] (label-one image-filename "models/resnet50.nippy"))
  ([image-filename trained-model]
   (let [data [{:input-1 (-> image-filename (io/file) (i/load-image) (i/resize 224 224)
                             (patch/image->patch :datatype :float :normalize false)
                             ;; https://github.com/fchollet/deep-learning-models/blob/master/imagenet_utils.py
                             (patch/patch-mean-subtract 103.939 116.779 123.68))}]]
     (->>
       (execute/run (util/read-nippy-file trained-model) data :batch-size 1)
       (first)
       :fc1000-activation
       (util/max-index)))))
