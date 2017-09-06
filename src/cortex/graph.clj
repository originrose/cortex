(ns cortex.graph
  "Several algorithms in cortex are simplified by using a simple directed graph structure.  There
  are at this point two different general classes of nodes and these are differentiated by
  understanding which pass they take part in.  All node's have a type and this type links
  to a metadata multimethod which gives further information on the node.  All nodes are functions
  taking a map of arguments.  Layers are functions which also have implicit input and output
  arguments which correspond to the edges of the graph the layers attach to."
  (:require [cortex.util :as util]
            [clojure.set :as c-set]
            [cortex.keyword-fn :as keyword-fn]
            [cortex.buffer-initialization :as buf-init]
            [clojure.core.matrix :as m]
            [cortex.argument :as arg]))


(defn- edges
  [graph]
  (get graph :edges))


(defn- edges->map
  [graph key-fn val-fn]
  (->> (edges graph)
       (group-by key-fn)
       (map (fn [[k v]]
              [k (map val-fn v)]))
       (into {})))


(defn parent->child-map
  [graph]
  (edges->map graph first second))


(defn child->parent-map
  [graph]
  (edges->map graph second first))


(defmulti get-node-metadata
  "Given that any node has a type member, return metadata on the node which
  must contain at least an :arguments member listing the arguments to the node."
  :type)

(defmethod get-node-metadata :default [node] {})


(defn get-node-argument-keys
  [node]
  (->> (get-node-metadata node)
       :arguments
       keys))


(defn get-node-metadata-arguments
  [node]
  (->> (get-node-metadata node)
       :arguments
       (map (fn [[k v]]
              (assoc v :key k)))))


(defn get-node-argument
  [node arg-key]
  (let [learn-atten (get node :learning-attenuation 1.0)
        non-trainable? (get node :non-trainable? false)
        node-defaults (:arguments (get-node-metadata node))
        default-arg (get node-defaults arg-key)
        default-arg (assoc default-arg :key arg-key)
        arg (util/deep-merge default-arg (get node arg-key))]
    (when-not default-arg
      (throw (ex-info (str "Invalid node argument: " arg-key)
                      {:node node
                       :arg arg-key
                       :accepted-args (get (get-node-metadata node) :arguments)})))
    (if (or (zero? (get arg :learning-attenuation learn-atten))
            non-trainable?)
      (assoc arg :gradients? false)
      arg)))


(defn get-node-arguments
  "Get the node arguments 'before' being merged with the node
buffers."
  [node]
  (->> (get-node-argument-keys node)
       (map #(get-node-argument node %))))


(defn empty-graph
  "Create an empty graph, which is stored as a map of:
  {:edges [] adjacency list of [id id]
   :id->node-map {} each node has an id and a type
   :buffers {} parameter buffers, map of id->{:buffer data :gradient gradient}
   :streams {} stream-name -> shape-descriptor.  Streams act as roots of the graph.
   }"
  []
  {:nodes   {}
   :edges   []
   :buffers {}
   :streams {}})


(defn get-node
  [graph node-id]
  (let [retval (get-in graph [:nodes node-id])]
    (when-not retval
      (throw (ex-info "Failed to find node:"
                      {:node-id node-id
                       :nodes (keys (get graph :nodes))})))
    retval))


(defn- get-or-create-node-id
  "Generate an id for this node."
  [graph node]
  (if-let [existing-id (get node :id)]
    (do
      (when-let [existing-node (get-in graph [:nodes existing-id])]
        (throw (ex-info "Duplicate id detected in graph:"
                        {:new-node node
                         :existing-node existing-node})))
      node)
    (assoc node :id (util/generate-id (name (get node :type))
                                      (set (keys (get graph :nodes)))))))


(defn add-node
  "Add a node to the graph with a list of predecessors.  If the node has no id one will
  be generated; if it does and it is not unique and exception will be thrown.
  If any of the predecessors does not exist an error will be thrown.  Returns a pair
  of [graph node-id]"
  [graph node predecessor-id-seq]
  (when-not (contains? graph :nodes)
    (throw (ex-info "nil graph in add-node"
                    {:graph graph
                     :node node})))
  (when-not (every? (get graph :nodes) predecessor-id-seq)
    (throw (ex-info "Failed to find all predecessor id's in graph"
                    {:id-seq predecessor-id-seq
                     :missing-ids (remove (get graph :nodes) predecessor-id-seq)
                     :existing-ids (vec (keys (get graph :nodes)))})))
  (let [node (get-or-create-node-id graph node)]
    [(-> graph
         (assoc-in [:nodes (get node :id)] node)
         (update :edges #(concat %
                                 (map vector
                                      predecessor-id-seq
                                      (repeat (get node :id))))))
     (get node :id)]))


(defn- recur-remove-node
  [p->c-map graph node-id]
  (let [graph (reduce (partial recur-remove-node p->c-map)
                      graph
                      (get p->c-map node-id))
        buffer-ids (->> (get-node-arguments (get-node graph node-id))
                        (filter #(= :parameter (get % :type)))
                        (map :buffer-id))]
    (-> graph
        (update :edges #(remove (fn [[p c]]
                                  (or (= p node-id)
                                      (= c node-id)))
                                %))
        (update :buffers #(apply dissoc % buffer-ids))
        (update :nodes dissoc node-id))))


(defn remove-node
  "Remove a node, its buffers and all children from the graph."
  [graph node-id]
  (recur-remove-node (parent->child-map graph) graph node-id))


(defn remove-children
  "Remove all children of a node (and there children, recursively)."
  [graph parent-node-id]
  (let [p->c-map (parent->child-map graph)]
    (reduce (partial recur-remove-node p->c-map)
            graph
            (get p->c-map parent-node-id))))



(defn any-trainable-arguments?
  [node]
  (->> (get-node-arguments node)
       (filter :gradients?)
       seq))


(defmulti build-node
  "Callback called when the node is added to the graph.  Note that the node at this point
  is not located in the graph.  Also note that any parameter arguments are generated
  in a separate step.  This is simply a translation from node->node called during
  the add-node step.  Predessors have been built, successors have not been built."
  (fn [graph node predecessor-ids successor-ids]
    (get node :type)))


(declare node->output-dimensions)


(defn clear-dimension-identifiers
  "Remove identifiers on dimensions.  This is used as a general tool when copying input
  dimensions to output dimensions.  The identifiers on the input dimensions should be removed."
  [dims]
  (dissoc dims :id :stream))


(defn ensure-single-output-dimensions
  "Ensure a node has one output and get it's output dimensions."
  [previous node]
  (let [output-dims (node->output-dimensions previous)]
   (when-not (or (= 1 (count output-dims))
                 (= 1 (count (filter #(= (get node :id)
                                         (get % :id))
                                     output-dims))))
     (throw (ex-info "Previous node has multiple output dimensions pertaining to this node."
                     {:previous previous
                      :node node
                      :output-dimensions output-dims})))
   (clear-dimension-identifiers (first output-dims))))



(defn carry-input-dims-forward
  [previous item]
  (assoc item :input-dimensions [(assoc
                                  (ensure-single-output-dimensions previous item)
                                  :id (get previous :id))]))

(defn carry-io-dims-forward
  [previous item]
  (let [input-dims (ensure-single-output-dimensions previous item)]
    ;;For single input nodes it is still important to note the parent this input came from.
    (assoc item :input-dimensions [(assoc input-dims :id (get previous :id))]
           ;;But for single outputs the source is clear.
           :output-dimensions [input-dims])))


(defn ensure-single-parent
  [graph node previous-id-seq]
  (when-not (= 1 (count previous-id-seq))
    (throw (ex-info "Node only takes a single node of input."
                    {:node node
                     :previous previous-id-seq})))
  (get-node graph (first previous-id-seq)))


(defn- default-build-fn
  [graph node predecessor-ids successor-ids]
  (let [previous (ensure-single-parent graph node predecessor-ids)]
    (carry-io-dims-forward previous node)))



;;lots of nodes do not need to be built.
(defmethod build-node :default
  [& args]
  (apply default-build-fn args))


(defn stream-descriptor
  "Shape descriptors are used to describe streams.  Currently there are two types
of streams, a multi-channeled input like an image and a single channel input like
a vector of floats."
  ([channels height width]
   {:channels channels
    :height height
    :width width})
  ([width]
   (stream-descriptor 1 1 width)))


(defn shape->stream-descriptor
  [shape]
  (condp = (count shape)
    1 (stream-descriptor (first shape))
    3 (let [[channels height width] shape]
        (stream-descriptor channels height width))))


(defn stream-descriptor->size
  ^long [shape-desc]
  (long (apply * (vals shape-desc))))


(defn add-stream
  [graph stream-name shape-descriptor]
  (assoc-in graph [:streams stream-name] shape-descriptor))


(defn stream->descriptor
  [graph stream-name]
  (if-let [stream-shape (get-in graph [:streams stream-name])]
    stream-shape
    (throw (ex-info "Failed to find stream in graph"
                    {:stream stream-name
                     :available-streams (keys (get graph :streams))}))))


(defn stream->size
  [graph stream-name]
  (stream-descriptor->size (stream->descriptor graph stream-name)))


(defn get-node
  [graph node-id]
  (if-let [node (get-in graph [:nodes node-id])]
    node
    (throw (ex-info (str "Failed to find node: " node-id (nil? node-id))
                    {:node-id node-id
                     :nodes (keys (:nodes graph))}))))


(defn- get-or-create-node-id
  "Generate an id for this node."
  [graph node]
  (if-let [existing-id (get node :id)]
    (do
      (when-let [existing-node (get-in graph [:nodes existing-id])]
        (throw (ex-info "Duplicate id detected in graph:"
                        {:new-node node
                         :existing-node existing-node})))
      node)
    (assoc node :id (util/generate-id (name (get node :type))
                                      (set (keys (get graph :nodes)))))))


(defn- edges
  [graph]
  (get graph :edges))


(defn- parent-seq
  [graph]
  (map first (edges graph)))


(defn- child-seq
  [graph]
  (map second (edges graph)))


(defn- parent-set
  [graph]
  (-> (parent-seq graph)
      set))


(defn- child-set
  [graph]
  (-> (child-seq graph)
      set))


(defn- set->ordered-vec
  [item-set item-seq]
  (->> (filter item-set item-seq)
       distinct
       vec))


(defn roots
  [graph]
  (-> (c-set/difference (parent-set graph) (child-set graph))
      (set->ordered-vec (parent-seq graph))))

(defn leaves
  [graph]
  (-> (c-set/difference (child-set graph) (parent-set graph))
      (set->ordered-vec (child-seq graph))))



(defn dfs-seq
  "Get a sequence of ids in dfs order."
  [graph]
  (let [p->c-map (-> (parent->child-map graph)
                     (assoc :roots (roots graph)))]

    (->> (tree-seq #(contains? p->c-map %)
                   #(get p->c-map %)
                   :roots)
         (drop 1)
         ;;Account for cases where the graph has multiple roots.  by taking the last occurance
         ;;of a multiply-occuring node.  this ensures that a child will not get visited until
         ;;after every parent has been visited.
         reverse
         distinct
         reverse)))


(defn relative-dfs-seq
  [graph node-id]
  (let [p->c-map (parent->child-map graph)]
    (tree-seq #(contains? p->c-map %)
              #(get p->c-map %)
              node-id)))


(defn leaf->root-seq
  [graph node-id]
  (let [c->p-map (-> (child->parent-map graph)
                     (assoc :roots [node-id]))]
    (->> (tree-seq #(contains? c->p-map %)
                   #(get c->p-map %)
                   :roots)
         (drop 1)
         (reverse)
         distinct
         (reverse))))


(defn create-node-dimensions
  "Create a node dimension map.  Dimensions are a map of
:channels :height :width with the last item (width) being the most
rapidly changing index and channels being the least rapidly changing index."
  ([channels height width]
   {:channels channels
    :height height
    :width width})
  ([width] (create-node-dimensions 1 1 width)))


(defn- node-inline-data->dims
  [node key-stem]
  (let [retval
        (if-let [width (get node (keyword (str key-stem "-width")))]
          [{:channels (get node (keyword (str key-stem "-channels")))
            :height (get node (keyword (str key-stem "-height")))
            :width width}]
          [{:channels 1
            :height 1
            :width (get node (keyword (str key-stem "-size")))}])]
    (when-not (every? number? (vals (first retval)))
      (throw (ex-info "Failed to convert node's built information into dimensions"
                      {:node node
                       :result retval
                       })))
    retval))

(defn node->input-dimensions
  "Return a list of dimensions in order, one for every input of the node."
  [node]
  (or (get node :input-dimensions)
      (node-inline-data->dims node "input")))


(defn node->output-dimensions
  "Return a list of dimensions in order, one for every input of the node."
  [node]
  (or (get node :output-dimensions)
      (node-inline-data->dims node "output")))


(defn dimensions->dims-with-ids
  "When there are only 1 of dimensions and ids then the id of the first dimension
is unambiguous and set into the dimension entry.  When there are more ensure
that each id has an unambiguous mapping to a dimension."
  [dims-seq id-seq]
  (when-not (= (count dims-seq) (count id-seq))
    (throw (ex-info "Dimensions vector and id sequence mismatch"
                    {:dims-sequence dims-seq
                     :id-sequence id-seq})))
  (if (= 1 (count dims-seq))
    (update (vec dims-seq) 0
            assoc :id (first id-seq))
    ;;Error check the dimensions that each id in the id seq appears in them
    ;;and that id's are not repeated.
    (do
      (when-not (every? #(contains? % :id)
                        dims-seq)
        (throw (ex-info "Every dimension entry must contain a mapping id"
                        {:dimensions dims-seq})))
      (let [dims-ids (set (map :id dims-seq))
            id-set (set id-seq)]
        (when-not (= (count id-seq)
                     (count id-set))
          (throw (ex-info "Duplicate ids detected"
                          {:id-set id-set
                           :id-seq id-seq
                           :dims-seq dims-seq})))
        (when-not (= id-set dims-ids)
          (throw (ex-info "id set differs from dimension id set"
                          {:id-seq id-seq
                           :dimension-seq dims-seq})))
        (vec dims-seq)))))


(defn node->output-dimension
  [node]
  (let [output-dims (node->output-dimensions node)]
    (when-not (= 1 (count output-dims))
      (throw (ex-info "Node has multiple outputs thus canonical dimension is ambiguous."
                      {:node node
                       :output-dims output-dims})))
    (first output-dims)))


(defn node->input-dimension
  [node]
  (let [input-dims (node->input-dimensions node)]
    (when-not (= 1 (count input-dims))
      (throw (ex-info "Node has multiple inputs thus canonical dimension is ambiguous"
                      {:node node
                       :input-dims input-dims})))
    (first input-dims)))


(defn dimensions->size
  ^long [dims]
  (apply * (vals (clear-dimension-identifiers dims))))


(defn dimensions->shape
  "Return a vector of integers with the highest indexes changing more rapidly than the
lower indexes...In other words the dimenion tuple is in big-endian order."
  [dims]
  (mapv dims [:channels :height :width]))


(defn- dims-vec->size
  ^long [node dims-vec direction]
  (when-not (= 1 (count dims-vec))
    (throw (ex-info "Cannot convert to size, node has multiple or zero dimensions"
                    {:node node
                     :direction direction})))
  (dimensions->size (first dims-vec)))


(defn node->input-size
  "Given a node, ensure it has 1 input dimension and call dimension->size for that dim."
  ^long [node]
  (dims-vec->size node (node->input-dimensions node) :input))


(defn node->output-size
  "Given a node, ensure it has 1 input dimension and call dimension->size for that dim."
  ^long [node]
  (dims-vec->size node (node->output-dimensions node) :output))


(defn- do-build-graph
  [c->p-map p->c-map graph node-id]
  (let [node (build-node graph (get-node graph node-id)
                         (get c->p-map node-id) (get p->c-map node-id))]
    (update graph :nodes assoc node-id node)))


(defn build-graph
  "Propagate size information (input/output sizes) through the graph in dfs order."
  [graph]
  (let [c->p-map (child->parent-map graph)
        p->c-map (parent->child-map graph)]
    (reduce (partial do-build-graph c->p-map p->c-map)
            graph
            (dfs-seq graph))))


(defn update-node
  [graph node-id update-fn]
  (when-not (contains? (get graph :nodes) node-id)
    (throw (ex-info "Update failed to find node"
                    {:node-id node-id})))
  (update-in graph [:nodes node-id] update-fn))


(defmulti get-argument-shape
  "Get the expected shape of an argument"
  (fn [graph node argument]
    (get argument :type)))


(defmethod get-argument-shape :stream
  [graph node argument]
  (if-let [retval (stream->size graph (get argument :stream))]
    [(long retval)]
    (throw (ex-info "Failed to find stream size for argument"
                    {:stream (get argument :stream)
                     :streams (keys stream->size)}))))


(defmethod get-argument-shape :node-output
  [graph node argument]
  (let [target-node (get-node graph (get argument :node-id))]
    (if-let [retval (node->output-size target-node)]
      [(long retval)]
      (throw (ex-info "Failed to find node output size"
                      {:argument argument
                       :nodes (keys (get graph :nodes))})))))

(defmethod get-argument-shape :node-argument
  [graph node argument]
  (let [target-node (get-node graph (get argument :node-id))
        target-arg (get-node-argument target-node (get argument :argument))]
    (when-not (and target-node target-arg)
      (throw (ex-info "Failed to find node or node argument"
                      {:node target-node
                       :argument target-arg
                       :src-argument argument})))
    (get-argument-shape graph target-node target-arg)))

(defmethod get-argument-shape :stream-augmentation
  [graph node argument]
  (throw (ex-info "Cannot get shape of stream augments without actually augmenting stream"
                  {:argument argument})))

(defmethod get-argument-shape :parameter
  [graph node argument]
  (try
    (keyword-fn/call-keyword-fn (get argument :shape-fn)
                                graph node argument)
    (catch Throwable e
      (throw (ex-info "Failed to resolve and call shape function"
                      {:node-id (get node :id)
                       :argument argument
                       :error e})))))

(defmulti initialize-graph-parameter-buffer
  "Initialize a graph parameter buffer"
  (fn
    [graph node argument shape initialization]
    (get initialization :type)))

(defmethod initialize-graph-parameter-buffer :default
  [graph node argument shape initialization]
  (buf-init/initialize-buffer (assoc initialization :shape shape)))


(defn- smart-shape-compare
  [shape1 shape2]
  (= (drop-while #(= 1 %) shape1)
     (drop-while #(= 1 %) shape2)))

(defn generate-parameter-argument-buffer
  "Given a parameter argument generate it's buffer."
  [node-id graph argument]
  (let [node (get-node graph node-id)
        expected-shape (get-argument-shape graph node argument)]
    (if-let [existing-buffer (get-in graph [:buffers (get argument :buffer-id) :buffer])]
      (do
        (when-not (smart-shape-compare expected-shape (m/shape existing-buffer))
          (throw (ex-info "Existing buffer does not match expected shape"
                          {:node-id node-id
                           :existing-shape (m/shape existing-buffer)
                           :expected-shape expected-shape
                           :node node})))
        graph)
      (let [param-buffer-id (util/generate-id (str (name (get node :id))
                                                   "-"
                                                   (name (get argument :key)))
                                              (set (keys (get graph :buffers))))
            param-buffer
            (if-let [user-supplied-buffer (get argument :buffer)]
              (let [user-shape (m/shape user-supplied-buffer)]
                (when-not (= user-shape expected-shape)
                  (throw (ex-info "User supplied buffer is incorrect shape"
                                  {:user-buffer-shape user-shape
                                   :expected-shape expected-shape})))
                user-supplied-buffer)
              (initialize-graph-parameter-buffer graph node argument
                                                 expected-shape
                                                 (get argument :initialization)))]
        (-> graph
            (assoc-in [:buffers param-buffer-id :buffer]
                      param-buffer)
            (update-in [:nodes node-id (get argument :key)]
                       dissoc :buffer)
            (update-in [:nodes node-id (get argument :key)]
                       assoc :buffer-id param-buffer-id))))))


(defn generate-node-parameter-buffers
  [graph node]
  (->> node
       get-node-arguments
       (filter #(= :parameter (get % :type)))
       (reduce (partial generate-parameter-argument-buffer (get node :id))
               graph)))

(defn generate-parameters
  "Go through all the nodes in the graph and generate any parameter buffers
that do not already exist.  Returns a new graph."
  [graph]
  (reduce
    (fn [graph id]
      (generate-node-parameter-buffers graph (get-node graph id)))
    graph
    (dfs-seq graph)))


(defn get-stream-augmentation-arguments
  [graph]
  (->> (dfs-seq graph)
       (map #(get-node graph %))
       (mapcat get-node-arguments)
       (filter #(= :stream-augmentation (get % :type)))))


(defn perform-stream-augmentations
  [stream-aug-args stream-map]
  (->> stream-aug-args
       (map (fn [{:keys [stream augmentation] :as arg}]
              (when-not (contains? stream-map stream)
                (throw (ex-info "Failed to find stream for augmentation"
                                {:argument arg
                                 :streams (vec (keys stream-map))})))
              (try
                (let [augment-result (keyword-fn/call-keyword-fn augmentation
                                                                 (get stream-map stream))
                      augment-data (if (get arg :datatype)
                                     {:datatype (get arg :datatype)
                                      :data augment-result}
                                     augment-result)]
                 [(arg/augmented-stream-arg->id arg) augment-data])
                (catch Throwable e
                  (throw (ex-info "Failed to augment stream"
                                  {:argument arg
                                   :error e}))))))
       (into {})
       (merge stream-map)))


(defn augment-streams
  "Augment the streams in the map and return a new map of data."
  [graph stream-map]
  (-> (get-stream-augmentation-arguments graph)
      (perform-stream-augmentations stream-map)))


(defmulti resolve-argument
  "Resolve a particular argument returning a map containing
at least :buffer if not both :buffer and :gradient."
  (fn [graph node argument stream-map node-id->output-map]
    (get argument :type)))


(defmethod resolve-argument :stream
  [graph node argument stream-map node-id->output-map]
  (if-let [buffer (get stream-map (get argument :stream))]
    buffer
    (throw (ex-info "Failed to resolve argument"
                    {:streams (keys stream-map)
                     :argument argument}))))

(defmethod resolve-argument :parameter
  [graph node argument stream-map node-id->output-map]
  ;;Rather than get-in here we use a function style lookup because
  ;;the buffers in the graph 'may' actually be a function instead
  ;;of a map to enable runtime systems to provide a minimal set of
  ;;parameters.
  (if-let [buffer ((get graph :buffers) (get argument :buffer-id))]
    buffer
    (throw (ex-info "Failed to resolve argument"
                    {:argument argument
                     :buffers (keys (get graph :buffers))}))))

(defmethod resolve-argument :node-output
  [graph node argument stream-map node-id->output-map]
  (if-let [buffer (get node-id->output-map (get argument :node-id))]
    buffer
    (throw (ex-info "Failed to resolve argument"
                    {:argument argument
                     :node-outputs (keys node-id->output-map)}))))

(defmethod resolve-argument :node-argument
  [graph node {:keys [node-id] :as argument} stream-map node-id->output-map]
  (let [target-node (get-node graph node-id)
        target-arg (get-node-argument target-node (get argument :argument))]
    (resolve-argument graph target-node target-arg stream-map node-id->output-map)))

(defmethod resolve-argument :stream-augmentation
  [graph node argument stream-map node-id->output-map]
  (if-let [buffer (get stream-map (arg/augmented-stream-arg->id argument))]
    buffer
    (throw (ex-info "Failed to resolve argument"
                    {:argument argument
                     :streams (keys stream-map)}))))


(defn resolve-arguments
  "Resolve the arguments to a particular node.  It is expected the stream map contains the
  augmented data if necessary.  Note that for uniformity the values are returned without
  modification.  This means the the format of the stream map and the node->output-map must be
  entries of the form of at least {:buffer data} instead of linking key directly to data.  This
  allows a uniform system both when doing auto-differentiation and when simply doing execution
  because when doing back propagation the entries must link to both {:buffer
  :gradient}."
  [graph node stream-map node-id->output-map]
  (->> (get-node-arguments node)
       (mapv (fn [{:keys [key type] :as argument}]
              [key (resolve-argument graph node argument
                                     stream-map node-id->output-map)]))
       (into {})))


(defn parameter-count
  "Return the number of trainable and non-trainable parameters."
  [graph]
  (->> (get graph :buffers)
       vals
       (map (comp #(apply * %) m/shape :buffer))
       (reduce +)))


(defn graph->nodes
  "Return a list of all nodes in the graph in dfs order."
  [graph]
  (->> (dfs-seq graph)
       (map #(get-node graph %))))


(defn get-parameter-buffer
  [graph buffer-id]
  (if-let [retval (get-in graph [:buffers buffer-id])]
    retval
    (throw (ex-info "Failed to find buffer for buffer id"
                    {:buffer-id buffer-id
                     :buffers (keys (get graph :buffers))}))))


(defmulti generate-stream-definitions
  "A stream definition is a pair of [stream shape].  Some nodes can generate this assuming
  they are built (normal loss terms for example) but most nodes cannot say anything useful
  for this step."
  (fn [graph node]
    (get node :type)))


(defmethod generate-stream-definitions :default
  [graph node]
  [])


(defn generate-leaf-streams
  "Given the graph datastructure, generate streams for stream bindings. Node's generate streams
  based on their output size and possibly based on some internal state defined in the node such
  as a combination of the stream they are bound to and another node's output size."
  [graph]
  (->> (leaves graph)
       (map #(get-node graph %))
       (mapcat (partial generate-stream-definitions graph))
       (reduce (fn [graph [stream shape]]
                 (add-stream graph stream (shape->stream-descriptor shape)))
               graph)))


(defn filter-graph
  "Given a graph and a predicate produce a new graph with only functions that match
the predicate."
  [graph pred]
  (->> (get graph :nodes)
       vals
       (remove pred)
       (map :id)
       (reduce remove-node graph)))


(defn graph->required-streams
  "Run through nodes of graph and keep track of streams encountered.  Return the set of stream
  names."
  [graph]
  (->> (get graph :nodes)
       vals
       (mapcat #(get-node-arguments %))
       (filter #(= :stream (get % :type)))
       (map :stream)
       set))


(defn graph->output-node-ids
  "Run through nodes of graph and identify nodes are either leaves or that are bound in
  arguments.  Only nodes that return truthy for predicate will be returned."
  [graph pred]
  (let [passing-leaves (->> (leaves graph)
                            (map #(get-node graph %))
                            (filter pred)
                            (map :id))
        ;;There is an inherent issue here in that losses (not loss gradients) are always
        ;;calculated on the cpu.  This means we have to jump through hoops in order to get them
        ;;evaulated during training (to measure loss).  This would be aleviated if loss terms
        ;;are made part of the graph.
        passing-args (->> (get graph :nodes)
                          vals
                          (mapcat #(get-node-arguments %))
                          (filter #(= :node-output (get % :type)))
                          (map :node-id)
                          (filter (comp pred #(get-node graph %))))]
    (set (concat passing-leaves passing-args))))
