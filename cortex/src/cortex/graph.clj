(ns cortex.graph
  "Several algorithms in cortex are simplified by using a simple directed graph structure.  There
are at this point two different general classes of nodes and these are differentiated by
understanding which pass they take part in.  All node's have a type and this type links
to a metadata multimethod which gives further information on the node.  All nodes are functions
taking a map of arguments.  Layers are functions which also have implicit input and output
  arguments which correspond to the edges of the graph the layers attach to."
  (:require [cortex.util :as util]
            [clojure.set :as c-set]
            [cortex.keyword-fn :as keyword-fn]))


(defmulti get-node-metadata
  "Given that any node has a type member, return metadata on the node which
  must contain at least an :arguments member listing the arguments to the node."
  :type)


(defmethod get-node-metadata :default [node] {})


(defn get-node-arguments
  "Get the node arguments 'before' being merged with the node
buffers."
  [node]
  (->> (-> (get-node-metadata node)
           (get :arguments {}))
       (map (fn [[arg-key arg-data]]
              (merge (assoc arg-data :key arg-key)
                     (get node arg-key {}))))))


(defmulti build-node
  "Callback called when the node is added to the graph.  Note that the node at this point
  is not located in the graph.  Also note that any parameter arguments are generated
  in a separate step.  This is simply a translation from node->node called during
  the add-node step."
  (fn [node graph predecessor-ids]
    (get node :type)))


(defrecord Graph [edges ;;Adjacency list of [id id]
                  id->node-map ;;each node has a :type
                  buffers ;;map of buffer-id->{:buffer data :gradient gradient}
                  ])


(defn create-graph
  []
  (->Graph [] {} {}))


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
  (->> (filter item-set ordered-item-seq)
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
  [graph ]
  (let [p->c-map (-> (parent->child-map graph)
                     (assoc :roots (roots graph)))]
    (->>
     (tree-seq #(contains? p->c-map %)
               #(get parent->child-map %)
               :roots)
     (drop 1))))


(defn- generate-parameter-argument-buffer
  [node-id graph argument]

  )


(defn- generate-parameter-buffers
  [graph id]
  (let [node (get-node graph id)
        arguments ]
    (->> (get-node-arguments node)
         (filter (= :parameter (get % :type)))
         (reduce (partial generate-parameter-argument-buffer id)
                 graph))))


(defn generate-parameters
  "Go through all the nodes in the graph and generate any parameter buffers
that do not already exist.  Returns a new graph."
  [graph]
  (->> (dfs-seq graph)
       (reduce generate-parameter-buffers
               graph)))
