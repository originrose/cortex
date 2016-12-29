(ns cortex.nn.traverse
  "Various graph traversal algorithms needed in order to implement
either inference or gradient descent on a layer graph.

Note that input-bindings are maps from node-id to stream
while output bindings are maps from node-id to {:stream :loss}."
  (:require [cortex.nn.build :as build]
            [cortex.nn.layers :as layers]
            [clojure.set :as c-set]
            [cortex.optimise :as optimise]))


(defn- check-node-id
  [network node-id]
  (when-not (get-in network [:layer-graph :id->node-map node-id])
    (throw (ex-info "Failed to find node id in graph"
                    {:node-id node-id
                     :graph-nodes (keys (get-in network [:layer-graph :id->node-map]))}))))


(defn bind-input
  "Create an input binding.  Inputs are always bound to incoming streams."
  [network node-id stream-name]
  (check-node-id network node-id)
  (assoc-in network [:traversal :input-bindings node-id] {:input-stream stream-name}))


(defn ->input-binding
  "Create a stand-alone input binding"
  [node-id stream-name]
  [node-id {:stream stream-name}])


(defn bind-input-bindings
  [network input-bindings]
  (reduce (fn [network [node-id {:keys [stream]}]]
            (bind-input network node-id stream))
          network
          input-bindings))


(defn bind-output-infer
  "Create an output binding.  For inference or to just get data out of the net
while training no stream or loss is necessary"
  [network node-id]
  (check-node-id network node-id)
  (assoc-in network [:traversal :output-bindings node-id {}]))


(defn bind-output-train
  "Bind an output for training which means the node has both a stream and a loss
  associated with it."
  [network node-id stream & [loss]]
  (check-node-id network node-id)
  (assoc-in network [:traversal :output-bindings node-id]
            {:output-stream stream
             :loss (or loss
                       (layers/auto-bind-loss node-id))}))

(defn ->output-binding
  "Create a stand-along output-binding"
  [node-id & {:keys [stream loss]}]
  [node-id {:stream stream
            :loss loss}])


(defn bind-output-bindings
  "Bind a list of output bindings to the network"
  [network output-bindings]
  (reduce (fn [network [node-id {:keys [stream loss]}]]
            (bind-output-train network node-id stream loss))
          network
          output-bindings))


(defn get-input-bindings
  [network]
  (->> (get-in network [:traversal :input-bindings])
       (map (fn [[node-id {:keys [input-stream]}]]
              {:node-id node-id
               :stream input-stream
               :direction :input}))))


(defn get-output-bindings
  [network]
  (->> (get-in network [:traversal :output-bindings])
       (map (fn [[node-id {:keys [output-stream loss]}]]
              {:node-id node-id
               :stream output-stream
               :direction :output
               :loss loss}))))


(defn auto-bind-io
  "Auto bind the network's roots and leaves to either :data :labels if there
are exactly 1 root and leaf or to :data-x where x is a one-based index of the
root and labels-x where labels are a 1-based index of the leaf.x"
  [network]
  (let [[roots leaves] (build/edges->roots-and-leaves (get-in network [:layer-graph :edges]))
        input-name-fn (if (> (count roots) 1)
                        (fn [network]
                          (keyword (str "data-" (+ 1 (count (get-input-bindings network))))))
                        (constantly :data))
        output-name-fn (if (> (count leaves) 1)
                         (fn [network]
                           (keyword (str "labels-" (+ 1 (count (get-output-bindings network))))))
                         (constantly :labels))]
    (as-> network network
      (reduce (fn [network root]
                (bind-input network root (input-name-fn network)))
              network
              roots)
      (reduce (fn [network leaf]
                (bind-output-train network leaf (output-name-fn network)))
              network
              leaves))))


(defn clear-io-bindings
  "Remove all io bindings (if any exist) from the network."
  [network]
  (update network :traversal
          (fn [traversal]
            (-> traversal
                (dissoc :input-bindings)
                (dissoc :output-bindings)))))


(defn get-io-bindings
  "get a sequence of maps of:
  {:node-id
  :stream
  :direction [:input :output]
  :loss-function (if output)}."
  [network]
  (concat (get-input-bindings network)
          (get-output-bindings network)))


(defn get-input-streams
  "Get all dataset streams used for input."
  [network]
  (->> (get-input-bindings network)
       (map :stream)
       distinct))


(defn get-output-streams
  "Get all dataset streams use for output"
  [network]
  (->> (get-output-bindings network)
       (map :stream)
       (remove nil?)
       distinct))


(defn get-output-training-bindings
  "Get a map of all of the output bindings that have both
stream and loss members."
  [network]
  (->> (get-output-bindings network)
       (filter #(and (get % :stream)
                     (get % :loss)))))


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


(defn- forward-traversal->buffer-map
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


(defn- clean-traversal-incoming-outgoing
  "Make the incoming and outgoing edges actually valid buffer keys
which means removing extra information from them."
  [traversal]
  (map (fn [entry]
         (-> entry
             (update :incoming #(map buffer-desc->map-key %))
             (update :outgoing #(map buffer-desc->map-key %))))
       traversal))


(defn- check-for-io-bindings
  "Without any io bindings there is no traversal."
  [network]
  (when-not (and (> (count (get-input-bindings network)) 0)
                 (> (count (get-output-bindings network)) 0))
    (throw (ex-info "Either no input or no output bindings were found on the network"
                    {:input-bindings (get-input-bindings network)
                     :output-bindings (get-output-bindings network)}))))


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
  [network
   & {:keys [optimiser]
      :or {optimiser (optimise/adam)}}]

  (check-for-io-bindings network)
  (let [forward-traversal (->> (create-forward-traversal network)
                               (filter-traversal network :training)
                               traversal->buffers)]
    (update network
            :traversal
            #(merge %
                    {:forward (clean-traversal-incoming-outgoing forward-traversal)
                     :backward (-> (reverse-forward-traversal forward-traversal)
                                   clean-traversal-incoming-outgoing)
                     :buffers (forward-traversal->buffer-map network forward-traversal)
                     :type :training
                     :optimiser optimiser}))))


(defn network->inference-traversal
  "Similar to network->gradient-descent however in this case we have the option
  of optimising for memory which means we can aggressively reuse buffers *or*
  optimising for speed in which case the result is the forward pass of gradient descent
  and we expect implementations to have multiple batches in flight simultaneously.  We
  default to optimising for memory because this avoids OOM situations with large networks."
  [{:keys [layer-graph] :as network}]
  (check-for-io-bindings network)
  (let [forward-traversal (->> (create-forward-traversal network)
                               (filter-traversal network :inference)
                               traversal->buffers)]
    (update network
            :traversal
            #(merge %
                    {:forward (clean-traversal-incoming-outgoing forward-traversal)
                     :buffers (forward-traversal->buffer-map network forward-traversal)
                     :type :inference}))))
