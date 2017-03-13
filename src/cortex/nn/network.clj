(ns cortex.nn.network
  "Translation from cortex layer vectors into actual network graphs."
  (:require
    [clojure.set :as c-set]
    [clojure.pprint :as pprint]
    [clojure.core.matrix :as m]
    [clojure.core.matrix.macros :refer [c-for]]
    [think.datatype.core :as dtype]
    [cortex.graph :as graph]
    [cortex.loss :as loss]
    [cortex.compute.driver :as drv]
    [cortex.compute.math :as math]
    [cortex.nn.layers :as layers]
    [cortex.nn.traverse :as traverse])
  (:import [java.util UUID]))


(defn embed-param-args
  [desc]
  (->> (graph/get-node-arguments desc)
       (filter #(= :parameter (get % :type)))
       (reduce (fn [desc argument]
                 (let [param-name (get argument :key)
                       node-param (get desc param-name)]
                   (if-not (map? node-param)
                     (assoc desc param-name {:buffer node-param})
                     desc)))
               desc)))


(defn- add-node-to-graph
  [[graph last-id] desc]
  (let [predecessor-id-seq (if (get desc :parents)
                             (get desc :parents)
                             (if last-id
                               [last-id]
                               []))
        ; TODO: Is this necessary?  For 1.0 why don't we get rid of
        ; any backward compatibility stuff, and then stabilize
        ; going forward?

        ;;For backward compatibility we need to embed any parameter argument
        ;;buffers into maps
        desc (embed-param-args desc)
        [graph id] (graph/add-node graph desc
                                   predecessor-id-seq)]
    [(if (= :input (get desc :type))
       (let [{:keys [output-channels
                     output-height
                     output-width]} desc]
         ; If we can generate the stream-descriptor from an input
         ; node, why store it in the graph?
         (-> graph
             (graph/update-node id #(assoc-in % [:input :stream] id))
             (graph/add-stream id
                               (graph/stream-descriptor
                                 output-channels
                                 output-height
                                 output-width))))
       graph)
     id]))


(defn network
  ([]
   {:compute-graph (graph/empty-graph)}))


(defn linear-network
  "Build the network, ensure the weights and biases are in place and of the
  appropriate sizes."
  ([network network-desc]
   (update network :compute-graph
     (fn [graph]
       (-> (first
             (reduce add-node-to-graph
                     [graph nil]
                     (flatten network-desc)))
           graph/build-graph
           graph/generate-parameters))))
  ([network-desc]
   (linear-network {:compute-graph (graph/empty-graph)} network-desc)))

(defn backend
  [network]
  (get-in network [:compute-binding :backend]))

(defn driver
  [network]
  (get-in network [:compute-binding :backend :driver]))

(defn stream
  [network]
  (get-in network [:compute-binding :backend :stream]))

(defn datatype
  [network]
  (get-in network [:compute-binding :backend :datatype]))

(defn parameters
  [network]
  (get-in network [:compute-binding :trainable-parameters]))

(defn optimizers
  [network]
  (get-in network [:compute-binding :optimizer]))

(defn loss-fn
  [network]
  (get-in network [:compute-binding :loss-function]))

;; When using these functions, make sure to call traverse/auto-bind-io
;; and traverse/network->training-traversal on the resulting network
(defn assoc-layers-to-network
  "Appends a list of layers to the end of the compute-graph"
  [network layer-list]
  (let [leaves (graph/leaves (get network :compute-graph))
        layer-list (vec (flatten layer-list))]
    (when-not (= 1 (count leaves))
      (throw (ex-info "cannot auto-append to graphs with either zero or multiple leaves"
                      {:leaves leaves})))
    (linear-network network (update layer-list 0 #(assoc % :parents leaves)))))


(defn dissoc-layers-from-network
  "Removes layers (nodes, edges, buffers) from the given parent node till the last leaf node"
  [network parent-node-id]
  (update network :compute-graph graph/remove-node parent-node-id))



(defn add-property-to-layer
  "Given a fully built network, adds properties like :learning-attenuation or :regularization to specific layers by node-id
  To get a list of node-id -> (keys (get-in network [:compute-graph :nodes)))
  ex: (add-property-to-layer network :conv-1 :learning-attentuation 0.0 :regularization )"
  [network node-id key value]
  (update network :compute-graph
          (fn [graph]
           (graph/update-node graph node-id #(assoc % key value)))))


(defn network->graph
  [network]
  (if-let [retval (get network :compute-graph)]
    retval
    (throw (ex-info "Network does not appear to contain a graph; keys should contain :compute-graph"
                    {:network-keys (keys network)}))))


(defn network->node
  [network node-id]
  (-> (network->graph network)
      (graph/get-node node-id)))


(defn network->node-parameters
  ([network node-id]
   (->> (graph/get-node (network->graph network) node-id)
        graph/get-node-arguments
        (filter #(= :parameter (get % :type)))
        (mapv (fn [arg]
                (merge arg
                       (graph/get-parameter-buffer (network->graph network)
                                                   (get arg :buffer-id)))))))
  ([network]
   (->> (network->graph network)
        graph/dfs-seq
        (mapcat (partial network->node-parameters network)))))


(defn loss-output-bindings
  [network]
  (->> (get-in network [:traversal :loss-function])
       (mapcat loss/get-loss-term-node-outputs)))


(defn output-bindings
  "Return the outputs of the network.  Anything explicity marked with an output binding
and anything that has a loss term attached to it's output becomes an output binding."
  [network]
  (let [forward-pass (get-in network [:traversal :forward])
        id->pass (->> (group-by :id forward-pass)
                      (map (fn [[k v]]
                             (when-not (= 1 (count v))
                               (throw (ex-info "Node mapped to multiple pass operations"
                                               {:node-id k
                                                :passes v})))
                             [k (first v)]))
                      (into {}))
        graph (network->graph network)]
    (->> (concat (traverse/get-output-bindings network)
                 (loss-output-bindings network))
         (map :node-id)
         distinct
         (map (fn [node-id]
                (when-not (= 1 (count (get-in id->pass [node-id :outgoing])))
                  (throw (ex-info "Output nodes must have a single output."
                                  {:node-id node-id
                                   :pass (get id->pass node-id)})))
                (let [output-id (first (get-in id->pass [node-id :outgoing]))]
                  {:node-id node-id
                   :buffers (get-in network [:compute-binding
                                             :traversal-buffers
                                             output-id])
                   :output-size (graph/node->output-size
                                 (graph/get-node graph node-id))}))))))


(defn input-bindings
  [network]
  (->> (traverse/get-input-bindings network)
       (filter #(get % :stream))
       (map (fn [{:keys [stream node-id] :as entry}]
              (assoc entry
                :buffers
                (get-in network [:compute-binding
                                 :traversal-buffers
                                 {:stream stream}])
                :size (get-in network [:compute-graph
                                       :nodes
                                       node-id
                                       :input-size]))))))


(defn output-values
  [{:keys [batch-size]} network]
  (->> output-bindings
       (map (fn [{:keys [buffers node-id output-size host-buffer elem-count]}]
              (let [buffer (get buffers :buffer)
                    double-buffers (->> (repeatedly batch-size
                                                    #(double-array output-size))
                                        vec)
                    copy-fn (fn [^long idx ^long output-size host-buffer double-buffers]
                              )]
                (drv/copy-device->host stream
                                       (math/device-buffer buffer) 0
                                       host-buffer 0
                                       elem-count)
                (drv/wait-for-event (drv/create-event stream))
                (if (< batch-size (.availableProcessors (Runtime/getRuntime)))
                  (c-for [idx 0 (< idx batch-size) (inc idx)]
                         (dtype/copy! host-buffer (long (* idx output-size))
                                      (get double-buffers idx) 0
                                      output-size))
                  (->> (pmap #(copy-fn % output-size host-buffer double-buffers)
                             (range batch-size))
                       dorun))
                [node-id double-buffers])))
       (into {})))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Print Layer Summary
(defn- network->parameter-keys
  [network]
  (->> network
       :compute-graph
       :nodes
       vals
       (mapcat graph/get-node-arguments)
       (filter #(= :parameter (get % :type)))
       (map :key)
       (distinct)
       (sort)))

(defn- print-graph-dimensions
  [dim-vec]
  (let [dims (first dim-vec)]
    (format "%sx%sx%s - %s"
            (:channels dims)
            (:height dims)
            (:width dims)
            (graph/dimensions->size dims))))

(defn- layer->input-str
  [layer]
  (print-graph-dimensions (graph/node->input-dimensions layer)))


(defn- layer->output-str
  [layer]
  (print-graph-dimensions (graph/node->output-dimensions layer)))


(defn- layer->buffer-shape
  [network layer k]
  (-> network
      (get-in [:compute-graph :buffers (get-in layer [k :buffer-id]) :buffer])
      m/shape))


(defn print-layer-summary
  "Given a network, prints a table summarizing layer input/output sizes as well
as parameter buffer shapes. This function does not work with descriptions (as
opposed to networks), but consider:

    (->> description
         network/linear-network
         traverse/auto-bind-io
         traverse/network->training-traversal
         network/print-layer-summary)"
  [network]
  (let [parameter-keys (network->parameter-keys network)]
    (->> network
         :traversal :forward
         (mapv (fn [{:keys [id]}]
                 (let [layer (graph/get-node (network->graph network) id)]
                   (into {"type" (:type layer)
                          "input" (layer->input-str layer)
                          "output" (layer->output-str layer)}
                         (for [k parameter-keys]
                                   [k (layer->buffer-shape network layer k)])))))
         (pprint/print-table (concat ["type" "input" "output"] parameter-keys))))
  ;(println "\nParameter count:" (graph/parameter-count (network->graph network)))
  )


(defn- node-id-is-in-pass?
  [graph pass node-id]
  (contains? (layers/get-pass-set
              (graph/get-node graph node-id))
             pass))


(defn leaf-inference-layers
  [network]
  (let [graph (network->graph network)
        is-inference? (partial node-id-is-in-pass? graph :inference)
        is-training? (partial node-id-is-in-pass? graph :training)
        keep-inference-nodes (fn [map-data]
                               (->> map-data
                                    (map (fn [[k v]]
                                           (when (is-inference? k)
                                             (when-let [v (-> (filter #(or (is-inference? %)
                                                                           (is-training? %)) v)
                                                             seq)]
                                              k))))
                                    (remove nil?)
                                    set))
        parent-set (keep-inference-nodes (graph/parent->child-map graph))
        child-set (keep-inference-nodes (graph/child->parent-map graph))]
    (c-set/difference child-set parent-set)))


