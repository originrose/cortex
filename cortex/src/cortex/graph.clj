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


(defmulti get-node-metadata
  "Given that any node has a type member, return metadata on the node which
  must contain at least an :arguments member listing the arguments to the node."
  :type)


(defmethod get-node-metadata :default [node] {})


(defn deep-merge
  "Like merge, but merges maps recursively.  Note that this is pulled from a rejected
patch to clojure.core: http://dev.clojure.org/jira/browse/CLJ-1468"
  [& maps]
  (if (every? map? maps)
    (apply merge-with deep-merge maps)
    (last maps)))


(defn get-node-argument
  [node arg-key]
  (let [retval (->> (get-node-metadata node)
                    (#(get % arg-key)))]
    (when-not retval
      (throw (ex-info "Failed to find node argument"
                      {:node node
                       :argument-name arg-key})))
    (let [retval (->> (assoc retval :key arg-key)
                      (deep-merge retval (get node arg-key)))]
      ;;Resolve the stream for the stream-augmentation argument dynamically
      ;;as late as possible.
      (if (= :stream-augmentation (get retval :type))
        (let [stream-arg (get-node-argument node (get retval :argument))]
          (when-not (= :stream (get stream-arg :type))
            (throw (ex-info "Stream augmentation arguments most point to stream arguments"
                            {:augmentation-arg retval
                             :target-arg stream-arg})))
          (assoc retval
                 :stream
                 (get stream-arg :stream)))
        retval))))


(defn get-node-arguments
  "Get the node arguments 'before' being merged with the node
buffers."
  [node]
  (->> (get-node-metadata node)
       :arguments
       keys
       (map #(get-node-argument node %))))


(defmulti build-node
  "Callback called when the node is added to the graph.  Note that the node at this point
  is not located in the graph.  Also note that any parameter arguments are generated
  in a separate step.  This is simply a translation from node->node called during
  the add-node step."
  (fn [node graph predecessor-ids]
    (get node :type)))


(defn create-graph
  "Create an empty graph."
  []
  {:edges [] ;;Adjacency list of [id id]
   :id->node-map ;;each node has an id and a type
   :buffers ;;parameter buffers, map of id->{:buffer data :gradient gradient}
   })


(defn get-node
  [graph node-id]
  (get-in graph [:id->node-map node-id]))


(defn- get-or-create-node-id
  "Generate an id for this node."
  [graph node]
  (if-let [existing-id (get node :id)]
    (do
      (when-let [existing-node (get-node graph existing-id)]
        (throw (ex-info "Duplicate id detected in graph:"
                        {:new-node node
                         :existing-node existing-node})))
      node)
    (assoc node :id (util/generate-id (name (get node :type))
                                      (set (keys (get graph :id->node-map)))))))


(defn add-node
  "Add a node to the graph with a list of predecessors.  If the node has no id one will
be generated; if it does and it is not unique and exception will be thrown.
If any of the predecessors does not exist an error will be thrown."
  [graph node predecessor-id-seq]
  (when-not (every? (get graph :id->node-map) predecessor-id-seq)
    (throw (ex-info "Failed to find all predecessor id's in graph"
                    {:id-seq predecessor-id-seq
                     :missing-ids (remove (get graph :id->node-map) predecessor-id-seq)
                     :existing-ids (vec (keys (get graph :id->node-map)))})))
  (let [node (-> (get-or-create-node-id graph node)
                 (build-node graph predecessor-id-seq))]
    (assoc graph
           :id->node-map node
           :edges (concat (get graph :edges)
                          (map vec
                               predecessor-id-seq
                               (repeat (get node :id)))))))

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

(defn- edges->map
  [graph key-fn val-fn]
  (->> (edges graph)
       (group-by key-fn)
       (map (fn [[k v]]
              [k (map val-fn v)]))
       (into {})))

(defn- parent->child-map
  [graph]
  (edges->map graph first second))

(defn- child->parent-map
  [graph]
  (edges->map graph second first))

(defn dfs-seq
  "Get a sequence of ids in dfs order."
  [graph]
  (let [p->c-map (-> (parent->child-map graph)
                     (assoc :roots (roots graph)))]
    (->>
     (tree-seq #(contains? p->c-map %)
               #(get parent->child-map %)
               :roots)
     (drop 1))))

(defmulti get-argument-shape
  "Get the expected shape of an argument"
  (fn [graph node argument stream->size]
    (get argument :type)))


(defmethod get-argument-shape :stream
  [graph node argument stream->size]
  (if-let [retval (get stream->size (get argument :stream))]
    [(long retval)]
    (throw (ex-info "Failed to find stream size for argument"
                    {:stream (get argument :stream)
                     :streams (keys stream->size)}))))

(defmethod get-argument-shape :node-output
  [graph node argument stream->size]
  (let [target-node (get-node graph (get argument :node-id))]
    (if-let [retval (get target-node :output-size)]
      [(long retval)]
      (throw (ex-info "Failed to find node output size"
                      {:argument argument
                       :nodes (keys (get graph :id->node-map))})))))

(defmethod get-argument-shape :node-argument
  [graph node argument stream->size]
  (let [target-node (get-node graph (get argument :node-id))
        target-arg (get-node-argument graph node (get argument :argument))]
    (get-argument-shape graph target-node target-arg stream->size)))

(defmethod get-argument-shape :stream-augmentation
  [graph node argument stream->size]
  (throw (ex-info "Cannot get shape of stream augments without actually augmenting stream"
                  {:argument argument})))

(defmethod get-argument-shape :parameter
  [graph node argument stream->size]
  (try
    (keyword-fn/call-keyword-fn (get argument :shape-fn)
                                graph node argument stream->size)
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


(defn- generate-parameter-argument-buffer
  "Given a parameter argument generate it's buffer."
  [node-id stream->size graph argument]
  (let [node (get-node graph node-id)
        expected-shape (get-argument-shape graph node argument stream->size)]
    (if-let [existing-buffer (get-in graph [:buffers (get argument :buffer-id) :buffer])]
      (do
        (when-not (= expected-shape (m/shape existing-buffer))
          (throw (ex-info "Existing buffer does not match expected shape"
                          {:node-id node-id
                           :existing-shape (m/shape existing-buffer)
                           :expected-shape expected-shape})))
        graph)
      (let [param-buffer-id (util/generate-id (str (name (get node :type))
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
            (assoc-in [:buffers param-buffer-id :buffer param-buffer])
            (update-in [:id->node-map node-id (get argument :key)] dissoc :buffer)
            (update-in [:id->node-map node-id (get argument :key)]
                       assoc :buffer-id param-buffer-id))))))


(defn- generate-parameter-buffers
  [stream->size graph id]
  (->> (get-node graph id)
       get-node-arguments
       (filter #(= :parameter (get % :type)))
       (reduce (partial generate-parameter-argument-buffer id stream->size)
               graph)))


(defn generate-parameters
  "Go through all the nodes in the graph and generate any parameter buffers
that do not already exist.  Returns a new graph."
  [graph stream->size]
  (reduce (partial generate-parameter-buffers stream->size)
          graph
          (dfs-seq graph)))


(defn augment-streams
  "Augment the streams in the map and return a new map of data."
  [graph stream-map]
  (->> (dfs-seq graph)
       (map #(get-node graph %))
       (mapcat get-node-arguments)
       (filter #(= :stream-augmentation (get % :type)))
       (map arg/augmented-stream-arg->id)
       (map (fn [{:keys [stream augmentation] :as arg}]
              (when-not (contains? stream-map stream)
                (throw (ex-info "Failed to find stream for augmentation"
                                {:argument arg
                                 :streams (vec (keys stream-map))})))
              (try
                [arg (keyword-fn/call-keyword-fn augmentation
                                                 (get stream-map stream))]
                (catch Throwable e
                  (throw (ex-info "Failed to augment stream"
                                  {:argument arg
                                   :error e}))))))
       (into {})
       (merge stream-map)))

(defmulti resolve-argument
  "Resolve a particular argument returning a map containing
at least :buffer if not both :buffer and :gradient."
  (fn [graph node argument stream-map node-id->output-map]
    (get argument :type)))


(defmethod resolve-argument :stream
  [graph node argument stream-map node-id->output-map]
  (if-let [buffer (get stream-map (get argument :stream))]
    (throw (ex-info "Failed to resolve argument"
                    {:streams (keys stream-map)
                     :argument argument}))))

(defmethod resolve-argument :parameter
  [graph node argument stream-map node-id->output-map]
  (if-let [buffer (get-in graph [:buffers (get argument :buffer-id)])]
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
  "Resolve the arguments to a particular node.
It is expected the stream map contains the augmented data if necessary.
Note that for uniformity the values are returned without modification.  This
means the the format of the stream map and the node->output-map must be
entries of the form of at least {:buffer data} instead of linking key directly
to data.  This allows a uniform system both when doing auto-differentiation and
when simply doing execution."
  [graph node stream-map node-id->output-map]
  (->> (get-node-arguments node)
       (map (fn [{:keys [key type] :as argument}]
              [key (resolve-argument graph node argument
                                     stream-map node-id->output-map)]))
       (into {})))
