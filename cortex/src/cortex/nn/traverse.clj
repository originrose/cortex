(ns cortex.nn.traverse
  "Various graph traversal algorithms needed in order to implement
either inference or gradient descent on a layer graph.

Note that input-bindings are maps from node-id to stream
while output bindings are maps from node-id to {:stream :loss}."
  (:require [cortex.nn.build :as build]
            [cortex.nn.layers :as layers]
            [clojure.set :as c-set]))


(defn- check-node-id
  [built-network node-id]
  (when-not (get-in built-network [:layer-graph :id->node-map node-id])
    (throw (ex-info "Failed to find node id in graph"
                    {:node-id node-id
                     :graph-nodes (keys (get-in built-network [:layer-graph :id->node-map]))}))))


(defn- bind-input
  "Bind a specific node to a dataset stream.  Returns a new network."
  [built-network node-id dataset-stream-name]
  ;;Multiple a node can bind to only one stream but one stream may be bound
  ;;to multiple nodes.
  (check-node-id built-network node-id)
  (assoc-in built-network [:traversal :input-bindings node-id] {:input-stream dataset-stream-name}))


(defn bind-input-bindings
  [built-network input-bindings]
  (reduce (fn [built-network  [id stream]]
            (bind-input built-network id stream))
          built-network
          input-bindings))


(defn- bind-output-infer
  "Enable output from a given graph node.  Since this is inference we do not need
  to worry about loss functions."
  [built-network node-id]
  (check-node-id built-network node-id)
  ;;Any node can produce an output stream but they have to be uniquely named.  A node
  ;;can bind to multiple output names.
  (assoc-in built-network [:traversal :output-bindings node-id] {}))


(defn- destructure-output-binding
  [output-binding]
  (if (keyword? output-binding)
    {:id output-binding}
    (let [[id {:keys [stream loss]}] output-binding]
      {:id id
       :output-stream stream
       :loss loss})))


(defn bind-output-inference-bindings
  [built-network output-bindings]
  (reduce (fn [built-network output-binding]
            (bind-output-infer built-network
                               (get (destructure-output-binding output-binding)
                                    :id)))
          built-network
          output-bindings))



(defn- bind-output-train
  "Enable output and bind it to a dataset stream for training.  If a loss function isn't specified
  then one will be chosen automatically based on the layer type."
  [built-network node-id dataset-stream-name & {:keys [loss-function]}]
  (check-node-id built-network node-id)
  (let [loss-function (if loss-function
                        loss-function
                        (layers/auto-bind-loss (get-in built-network
                                                       [:layer-graph :id->node-map node-id])))]
    (assoc-in built-network [:traversal :output-bindings node-id] {:output-stream dataset-stream-name
                                                                   :loss loss-function})))


(defn bind-output-training-bindings
  [built-network output-bindings]
  (reduce (fn [built-network output-binding]
            (let [{:keys [id output-stream loss]} (destructure-output-binding output-binding)]
              (bind-output-train built-network id output-stream :loss-function loss)))
          built-network
          output-bindings))


(defn get-dataset-bindings
  "get a sequence of maps of:
  {:node-id
  :dataset-stream
  :direction [:input :output]
  :loss-function (if output)}."
  [{:keys [traversal]}]
  (let [{:keys [input-bindings output-bindings]} traversal]
    (concat (map (fn [[node-id {:keys [input-stream]}]]
                   {:node-id node-id
                    :dataset-stream input-stream
                    :direction :input})
                 input-bindings)
            (map (fn [[node-id {:keys [output-stream loss]}]]
                   {:node-id node-id
                    :dataset-stream output-stream
                    :direction :output
                    :loss-function loss})
                 output-bindings))))



(defn create-forward-traversal
  "A forward traversal is a linear dfs order sequence.
There is an optional argument to remove nodes of a particular type from
the traversal.

Each item in the sequence is a map of:
{:incoming ()
 :id
 :outgoing ()
}"
  [{:keys [layer-graph] :as built-network}]
  (let [{:keys [id->node-map edges]} layer-graph
        {:keys [input-bindings output-bindings]} (get built-network :traversal)
        ;;Remove all edges that do not participate in the keep node set.
        [roots leaves] (build/edges->roots-and-leaves edges)
        parent->child-map (build/edges->parent->child-map edges)
        child->parent-map (build/edges->child->parent-map edges)]
    (->> (build/edges->dfs-seq edges :roots parent->child-map)
         (drop 1)
         (map (fn [id]
                {:incoming (concat
                            (->> [id]
                                 (map input-bindings)
                                 (remove nil?))
                            (->> (get child->parent-map id)
                                 (map (fn [id]
                                        {:id id}))))
                 :id id
                 :outgoing (concat (->> (get parent->child-map id)
                                        (map (fn [id] {:id id})))
                                   (->> [id]
                                        (map (fn [id]
                                               (when-let [output-binding (get output-bindings id)]
                                                 (merge {:output-id id}
                                                        output-binding))))
                                        (remove nil?)))})))))


(defn filter-traversal
  [{:keys [layer-graph]} pass-type traversal]
  (->> traversal
       (reduce (fn [[traversal input-alias-map] {:keys [incoming id] :as entry}]
                 (let [graph-node (get-in layer-graph [:id->node-map id])
                       pass-set (layers/get-pass-set graph-node)
                       new-incoming (flatten (map #(get input-alias-map (get % :id) %) incoming))]
                   (if (contains? pass-set pass-type)
                     [(conj traversal
                            (assoc entry
                                   :incoming new-incoming))
                      input-alias-map]
                     [(conj traversal entry) (assoc input-alias-map id new-incoming)])))
               [[] {}])
       first
       reverse
       (reduce (fn [[traversal output-alias-map] {:keys [id outgoing] :as entry}]
                 (let [graph-node (get-in layer-graph [:id->node-map id])
                       pass-set (layers/get-pass-set graph-node)
                       new-outgoing (flatten (map #(get output-alias-map (get % :id) %) outgoing))]
                   (if (contains? pass-set pass-type)
                     [(conj traversal
                            (assoc entry
                                   :outgoing new-outgoing))
                      output-alias-map]
                     [traversal (assoc output-alias-map id new-outgoing)])))
               [[] {}])
       first
       reverse))


(defn traversal->buffers
  "Traversals initial hold id of incoming nodes.  For the next steps
we need the incoming and outgoing edges to hold unique ids such that
the incoming buffer of the next step points to the outgoing buffer of
the previous step."
  [traversal]
  (->> traversal
       (reduce (fn [[traversal buffer-map] {:keys [incoming id outgoing] :as entry}]
                 [(conj traversal
                        {:incoming (flatten
                                    (map (fn [incoming-data]
                                           (if-let [id (get incoming-data :id)]
                                             (get buffer-map id)
                                             incoming-data))
                                         incoming))
                         :id id
                         :outgoing outgoing})
                  (assoc buffer-map id outgoing)])
               [[] {}])
       first))


(defn- reverse-forward-traversal
  "See create-forward-traversal.  Reverse of same sequence."
  [forward-traversal]
  (->> forward-traversal
       reverse
       (map (fn [{:keys [incoming outgoing] :as traverse-item}]
              (assoc traverse-item
                     :incoming outgoing
                     :outgoing incoming)))))


(defn- buffer-desc->map-key
  [buffer-desc]
  (select-keys buffer-desc [:id :input-stream :output-id]))


(defn forward-traversal->buffer-map
  [built-network forward-traversal]
  (let [id->node-map (get-in built-network [:layer-graph :id->node-map])]
    (reduce (fn [buffer-map {:keys [incoming id outgoing]}]
              (let [node (get id->node-map id)
                    output-size (get node :output-size)
                    input-size (get node :input-size)]
                (merge buffer-map
                       (->> (concat (map #(assoc % :size output-size) outgoing)
                                    (map #(assoc % :size input-size) incoming))
                            (map (fn [buffer-desc]
                                   [(buffer-desc->map-key buffer-desc) buffer-desc]))
                            (into {})))))
            {}
            forward-traversal)))


(defn clean-traversal-incoming-outgoing
  [traversal]
  (map (fn [entry]
         (-> entry
             (update :incoming #(map buffer-desc->map-key %))
             (update :outgoing #(map buffer-desc->map-key %))))
       traversal))


(defn network->training-traversal
  "Given network create master buffer list,
two traversals (forward,backward)
and input and output buffer lists.

!!Note that input-bindings are maps from stream to node-id
while output-bindings are maps from node-id to {:stream :loss}!!


Each traversal is sequence of maps like in create-forward-traversal
exception the incoming and outgoing ids are buffer ids.
Input bindings are pairs of node to stream name.  Output bindings
for gradient descent are also pairs of node-id to stream name or they can be
pairs of node-id to [stream-name loss-function].
{:buffers map id->{:size}
 :forward where incoming/outgoing maps to buffer id
 :backward where incoming/outgoing maps to buffer id}"
  [built-network input-bindings output-bindings
   & {:keys [optimiser]
      :or {optimiser (layers/adam)}}]
  (let [built-network (-> built-network
                          (bind-input-bindings input-bindings)
                          (bind-output-training-bindings output-bindings))
        forward-traversal (->> (create-forward-traversal built-network)
                               (filter-traversal built-network :training)
                               traversal->buffers)]
    (update built-network
            :traversal
            #(merge %
                    {:forward (clean-traversal-incoming-outgoing forward-traversal)
                     :backward (-> (reverse-forward-traversal forward-traversal)
                                   clean-traversal-incoming-outgoing)
                     :buffers (forward-traversal->buffer-map built-network forward-traversal)
                     :type :training
                     :optimiser optimiser}))))


(defn network->inference-traversal
  "Similar to network->gradient-descent however in this case we have the option
  of optimising for memory which means we can aggressively reuse buffers *or*
  optimising for speed in which case the result is the forward pass of gradient descent
  and we expect implementations to have multiple batches in flight simultaneously.  We
  default to optimising for memory because this avoids OOM situations with large networks."
  [{:keys [layer-graph] :as built-network}
   input-bindings output-bindings]
  (let [built-network (-> built-network
                          (bind-input-bindings input-bindings)
                          (bind-output-training-bindings output-bindings))
        forward-traversal (->> (create-forward-traversal built-network)
                               (filter-traversal built-network :inference)
                               traversal->buffers)]
    (update built-network
            :traversal
            #(merge %
                    {:forward (clean-traversal-incoming-outgoing forward-traversal)
                     :buffers (forward-traversal->buffer-map built-network forward-traversal)
                     :type :inference}))))
