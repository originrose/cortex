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
  (let [learn-atten (get node :learning-attenuation 1.0)
        non-trainable? (get node :non-trainable? false)
        retval (->> (get-node-metadata node)
                    :arguments
                    (#(get % arg-key)))]
    (when-not retval
      (throw (ex-info "Failed to find node argument"
                      {:node node
                       :argument-name arg-key
                       :arguments (get (get-node-metadata node) :arguments)})))
    (let [retval (->> (assoc retval :key arg-key)
                      (deep-merge retval (get node arg-key)))
          param-learn-atten (get retval :learning-attenuation learn-atten)]
      (if (or (zero? param-learn-atten)
              non-trainable?)
        (assoc retval :gradients? false)
        (assoc retval :learning-attenuation param-learn-atten)))))


(defn get-node-arguments
  "Get the node arguments 'before' being merged with the node
buffers."
  [node]
  (->> (get-node-metadata node)
       :arguments
       keys
       (map #(get-node-argument node %))))


(defn any-trainable-arguments?
  [node]
  (->> (get-node-arguments node)
       (filter :gradients?)
       seq))


(defmulti build-node
  "Callback called when the node is added to the graph.  Note that the node at this point
  is not located in the graph.  Also note that any parameter arguments are generated
  in a separate step.  This is simply a translation from node->node called during
  the add-node step."
  (fn [graph node predecessor-ids]
    (get node :type)))

;;lots of nodes do not need to build built.
(defmethod build-node :default
  [graph node p-id-seq]
  node)


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


(defn stream-descriptor->size
  ^long [shape-desc]
  (long (apply * (vals shape-desc))))


(defn empty-graph
  "Create an empty graph, which is stored as a map of:
  {:edges [] adjacency list of [id id]
   :id->node-map {} each node has an id and a type
   :buffers {} parameter buffers, map of id->{:buffer data :gradient gradient}
   :streams {} stream-name -> shape-descriptor.  Streams act as roots of the graph.
   }"
  []
  {:edges []
   :id->node-map {}
   :buffers {}
   :streams {}
   })


(defn add-stream
  [graph stream-name shape-descriptor]
  (assoc-in graph [:streams stream-name] shape-descriptor))


(defn stream->size
  [graph stream-name]
  (if-let [stream-shape (get-in graph [:streams stream-name])]
    (stream-descriptor->size stream-shape)
    (throw (ex-info "Failed to find stream in graph"
                    {:stream stream-name
                     :available-streams (keys (get graph :streams))}))))

(defn input-node
  [stream-name]
  {:type :input
   :input {:stream stream-name}})


(defmethod build-node :input
  [graph node predecessor-seq]
  (when-not (= 0 (count predecessor-seq))
    (throw (ex-info "Input nodes cannot have predecessor nodes"
                    {:node node
                     :predecessors predecessor-seq})))
  (let [input-data (get-node-argument node :input)]
    (when-not (= :stream (get input-data :type))
      (throw (ex-info "Input nodes can only link to streams"
                      {:node node})))
    (if-let [stream-desc (get-in graph [:streams (get input-data :stream)])]
      (let [channels (long (get stream-desc :channels))
            width (long (get stream-desc :width))
            height (long (get stream-desc :height))]
       (assoc node
              :input-channels channels
              :output-channels channels
              :input-height height
              :output-height height
              :input-width width
              :output-width width
              :input-size (* channels width height)
              :output-size (* channels width height)))
      (throw (ex-info "Failed to find stream to bind to input"
                      {:node node
                       :stream (get input-data :stream)})))))


(defmethod get-node-metadata :input
  [node]
  {:arguments {:input {:type :stream}}})


(defn get-node
  [graph node-id]
  (let [retval (get-in graph [:id->node-map node-id])]
    (when-not retval
      (throw (ex-info "Failed to find node:"
                      {:node-id node-id
                       :nodes (keys (get graph :id->node-map))})))
    retval))


(defn- get-or-create-node-id
  "Generate an id for this node."
  [graph node]
  (if-let [existing-id (get node :id)]
    (do
      (when-let [existing-node (get-in graph [:id->node-map existing-id])]
        (throw (ex-info "Duplicate id detected in graph:"
                        {:new-node node
                         :existing-node existing-node})))
      node)
    (assoc node :id (util/generate-id (name (get node :type))
                                      (set (keys (get graph :id->node-map)))))))


(defn add-node
  "Add a node to the graph with a list of predecessors.  If the node has no id one will
  be generated; if it does and it is not unique and exception will be thrown.
  If any of the predecessors does not exist an error will be thrown.  Returns a pair
  of [graph node-id]"
  [graph node predecessor-id-seq]
  (when-not (every? (get graph :id->node-map) predecessor-id-seq)
    (throw (ex-info "Failed to find all predecessor id's in graph"
                    {:id-seq predecessor-id-seq
                     :missing-ids (remove (get graph :id->node-map) predecessor-id-seq)
                     :existing-ids (vec (keys (get graph :id->node-map)))})))
  (let [node (get-or-create-node-id graph node)]
    [(-> graph
         (assoc-in [:id->node-map (get node :id)] node)
         (update :edges #(concat %
                                 (map vector
                                      predecessor-id-seq
                                      (repeat (get node :id))))))
     (get node :id)]))

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

(defn parent->child-map
  [graph]
  (edges->map graph first second))

(defn child->parent-map
  [graph]
  (edges->map graph second first))

(defn dfs-seq
  "Get a sequence of ids in dfs order."
  [graph]
  (let [p->c-map (-> (parent->child-map graph)
                     (assoc :roots (roots graph)))]
    (->>
     (tree-seq #(contains? p->c-map %)
               #(get p->c-map %)
               :roots)
     (drop 1))))

(defn relative-dfs-seq
  [graph node-id]
  (let [p->c-map (parent->child-map graph)]
    (tree-seq #(contains? p->c-map %)
              #(get p->c-map %)
              node-id)))


(defn- do-build-graph
  [c->p-map graph node-id]
  (let [node (build-node graph (get-node graph node-id) (get c->p-map node-id))]
    (update graph :id->node-map assoc node-id node)))


(defn build-graph
  "Propagate size information (input/output sizes) through the graph in dfs order."
  [graph]
  (let [c->p-map (child->parent-map graph)]
    (reduce (partial do-build-graph c->p-map)
            graph
            (dfs-seq graph))))


(defn update-node
  [graph node-id update-fn]
  (when-not (contains? (get graph :id->node-map) node-id)
    (throw (ex-info "Update failed to find node"
                    {:node-id node-id})))
  (update-in graph [:id->node-map node-id] update-fn))


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
    (if-let [retval (get target-node :output-size)]
      [(long retval)]
      (throw (ex-info "Failed to find node output size"
                      {:argument argument
                       :nodes (keys (get graph :id->node-map))})))))

(defmethod get-argument-shape :node-argument
  [graph node argument]
  (let [target-node (get-node graph (get argument :node-id))
        target-arg (get-node-argument graph node (get argument :argument))]
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


(defn- generate-parameter-argument-buffer
  "Given a parameter argument generate it's buffer."
  [node-id graph argument]
  (let [node (get-node graph node-id)
        expected-shape (get-argument-shape graph node argument)]
    (if-let [existing-buffer (get-in graph [:buffers (get argument :buffer-id) :buffer])]
      (do
        (when-not (= expected-shape (m/shape existing-buffer))
          (throw (ex-info "Existing buffer does not match expected shape"
                          {:node-id node-id
                           :existing-shape (m/shape existing-buffer)
                           :expected-shape expected-shape})))
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
            (update-in [:id->node-map node-id (get argument :key)]
                       dissoc :buffer)
            (update-in [:id->node-map node-id (get argument :key)]
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


(defn augment-streams
  "Augment the streams in the map and return a new map of data."
  [graph stream-map]
  (->> (dfs-seq graph)
       (map #(get-node graph %))
       (mapcat get-node-arguments)
       (filter #(= :stream-augmentation (get % :type)))
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
  "Resolve the arguments to a particular node.
It is expected the stream map contains the augmented data if necessary.
Note that for uniformity the values are returned without modification.  This
means the the format of the stream map and the node->output-map must be
entries of the form of at least {:buffer data} instead of linking key directly
to data.  This allows a uniform system both when doing auto-differentiation and
when simply doing execution because when doing back propagation the entries must
link to both {:buffer :gradient}."
  [graph node stream-map node-id->output-map]
  (->> (get-node-arguments node)
       (map (fn [{:keys [key type] :as argument}]
              [key (resolve-argument graph node argument
                                     stream-map node-id->output-map)]))
       (into {})))

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
        (update :id->node-map dissoc node-id))))


(defn remove-children
  "Remove all child nodes, edges and any associated buffers from the graph."
  [graph parent-node-id]
  (let [p->c-map (parent->child-map graph)]
    (reduce (partial recur-remove-node p->c-map)
            graph
            (get p->c-map parent-node-id))))

(defn remove-node
  "Remove a node, its buffers and all children from the graph."
  [graph node-id]
  (recur-remove-node (parent->child-map graph) graph node-id))


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
