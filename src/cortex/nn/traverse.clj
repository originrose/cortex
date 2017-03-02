(ns cortex.nn.traverse
  "Various graph traversal algorithms needed in order to implement
either inference or gradient descent on a layer graph.

Note that input-bindings are maps from node-id to stream
while output bindings are maps from node-id to {:stream :loss}."
  (:require [cortex.nn.network :as network]
            [cortex.nn.layers :as layers]
            [cortex.graph :as graph]
            [clojure.set :as c-set]
            [cortex.optimize :as optimize]
            [cortex.optimize.adam :as adam]
            [cortex.loss :as loss]
            [cortex.core-matrix-backends :as b]
            [clojure.core.matrix :as m]
            [cortex.buffer-initialization :as buf-init]
            [cortex.argument :as arg]))


(defn input-binding
  "Create a stand-alone input binding"
  [node-id stream-name]
  [node-id {:stream stream-name}])


(defn output-binding
  "Create a stand-alone output-binding"
  [node-id & {:keys [stream loss]}]
  [node-id {:stream stream
            :loss loss}])


(defn- check-node-id
  "Check whether a given node-id is in the network."
  [network node-id]
  (when-not (get-in network [:layer-graph :id->node-map node-id])
    (throw (ex-info "Failed to find node id in graph"
                    {:node-id node-id
                     :graph-nodes (keys (get-in network [:layer-graph :id->node-map]))}))))


(defn bind-input-to-stream
  "Create an input binding.  Inputs are always bound to incoming streams."
  [network node-id stream-name]
  (check-node-id network node-id)
  (assoc-in network [:traversal :input-bindings node-id] {:stream stream-name}))


(defn bind-input-bindings
  [network input-bindings]
  (reduce (fn [network [node-id {:keys [stream]}]]
            (bind-input-to-stream network node-id stream))
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
            {:stream stream
             :loss (or loss
                       (layers/get-layer-default-loss
                        (graph/get-node (get network :layer-graph) node-id)))}))


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
       (map (fn [[node-id {:keys [stream]}]]
              {:node-id node-id
               :stream stream
               :direction :input}))))


(defn get-output-bindings
  [network]
  (->> (get-in network [:traversal :output-bindings])
       (map (fn [[node-id {:keys [stream loss]}]]
              {:node-id node-id
               :stream stream
               :direction :output
               :loss loss}))))


(defn clear-io-bindings
  "Remove all io bindings (if any exist) from the network."
  [network]
  (update network :traversal
          (fn [traversal]
            (-> traversal
                (dissoc :input-bindings)
                (dissoc :output-bindings)))))

(declare remove-existing-loss-terms)

(defn bind-vars-to-network
  "Bind network nodes to dataset variables."
  [network]
  (let [network (clear-io-bindings network)
        ;;Get the graph without any loss terms else we will bind things to the loss nodes.
        graph (get (remove-existing-loss-terms network) :layer-graph)
        inputs (graph/roots graph)
        outputs (graph/leaves graph)

        ; Bind the inputs
        network (reduce (fn [network node-id]
                          (bind-input-to-stream network node-id node-id))
                        network
                        inputs)
        ; Bind the outputs
        network (reduce (fn [network node-id]
                          (bind-output-train network node-id node-id))
                        network
                        outputs)]
    network))

(defn auto-bind-io
  "Auto bind the network's roots and leaves to either :data :labels if there
are exactly 1 root and leaf or to :data-x where x is a one-based index of the
root and labels-x where labels are a 1-based index of the leaf."
  [network]
  (let [network (clear-io-bindings network)
        ;;Get the graph without any loss terms else we will bind things to the loss nodes.
        graph (get (remove-existing-loss-terms network) :layer-graph)
        inputs (graph/roots graph)
        outputs (graph/leaves graph)
        input-name-fn (if (> (count inputs) 1)
                        (fn [network]
                          (keyword (str "data-" (+ 1 (count (get-input-bindings network))))))
                        (constantly :data))
        output-name-fn (if (> (count outputs) 1)
                         (fn [network]
                           (keyword (str "labels-" (+ 1 (count (get-output-bindings network))))))
                         (constantly :labels))
        ; Bind the inputs
        network (reduce (fn [network root]
                          (bind-input-to-stream network root (input-name-fn network)))
                        network
                        inputs)
        ; Bind the outputs
        network (reduce (fn [network leaf]
                          (bind-output-train network leaf (output-name-fn network)))
                        network
                        outputs)]
    network))


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
  [{:keys [layer-graph] :as network}]
  (let [{:keys [input-bindings output-bindings]} (get network :traversal)
        ;;Remove all edges that do not participate in the keep node set.
        child->parent-map (graph/child->parent-map layer-graph)
        output-bindings (->> output-bindings
                             (map (fn [[k v]]
                                    [k (dissoc v :stream)]))
                             (into {}))]
    (->> (graph/dfs-seq layer-graph)
         (reduce (fn [[retval id->buffer-map] id]
                   (let [node-buffer (if-let [output-binding (get output-bindings id)]
                                       (merge {:output-id id} output-binding)
                                       {:id id})]
                     [(conj retval {:incoming (concat
                                               (->> [id]
                                                    (map input-bindings)
                                                    (remove nil?))
                                               (->> (get child->parent-map id)
                                                    (map (fn [id]
                                                           (get id->buffer-map id)))))
                                    :id id
                                    :outgoing [node-buffer]})
                      (assoc id->buffer-map id node-buffer)]))
                 [[] {}])
         first)))


(defn filter-traversal
  [{:keys [layer-graph]} pass-type traversal]
  (->> traversal
       (reduce (fn [[traversal input-alias-map] {:keys [incoming id] :as entry}]
                 (let [graph-node (graph/get-node layer-graph id)
                       pass-set (layers/get-pass-set graph-node)
                       new-incoming (flatten (map #(get input-alias-map (get % :id) %)
                                                  incoming))]
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
                       new-outgoing (flatten (map #(get output-alias-map
                                                        (get % :id) %) outgoing))]
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
  [traversal buffer-map]
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
               [[] buffer-map])))


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
  (select-keys buffer-desc [:id :stream :output-id]))


(defn- forward-traversal->buffer-map
  [network forward-traversal]
  (let [layer-graph (get network :layer-graph)]
    (reduce (fn [buffer-map {:keys [incoming id outgoing]}]
              (let [node (graph/get-node layer-graph id)
                    output-size (get node :output-size)
                    input-size (get node :input-size)]
                (when-not (and output-size input-size)
                  (throw (ex-info "Node does not have input or output size"
                                  {:node node})))
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


(defn- remove-non-trainable
  [network traversal]
  (-> (reduce (fn [[keep-set traversal] {:keys [incoming id] :as item}]
                (let [keep? (or (seq (filter #(contains? keep-set (get % :id)) incoming))
                                (graph/any-trainable-arguments? (graph/get-node
                                                                 (get  network :layer-graph)
                                                                  id)))]
                  (if keep?
                    [(conj keep-set id) (conj traversal item)]
                    [keep-set traversal])))
              [#{} []]
              traversal)
      second))


(defn- check-for-io-bindings
  "Without any io bindings there is no traversal."
  [network]
  (when-not (and (> (count (get-input-bindings network)) 0)
                 (> (count (get-output-bindings network)) 0))
    (throw (ex-info "Either no input or no output bindings were found on the network"
                    {:input-bindings (get-input-bindings network)
                     :output-bindings (get-output-bindings network)}))))


(defn remove-existing-loss-terms
  [network]
  (-> network
      (update :traversal dissoc :loss-function)
      (update :layer-graph
              #(->> (graph/dfs-seq %)
                    (map (fn [node-id]
                           (let [pass-set (-> (graph/get-node % node-id)
                                              graph/get-node-metadata
                                              :passes
                                              set)]
                             (when (contains? pass-set :loss)
                               node-id))))
                    (remove nil?)
                    (reduce graph/remove-node %)))))


(defn- map->loss-term-seq
  [item-map]
  (->> (keys item-map)
       (map (fn [loss-key]
              (loss/loss-term-from-map-key-val loss-key (get item-map loss-key))))
       (remove nil?)))


(defn- generate-node-loss-terms
  [network node]
  (let [node-losses (->> (map->loss-term-seq node)
                         (map #(arg/set-arg-node-output % :output (get node :id))))
        trainable-parameters (->> (graph/get-node-arguments node)
                                  (filter :gradients?))
        parameter-losses (->> trainable-parameters
                              (map map->loss-term-seq)
                              (mapcat (fn [parameter loss-term-seq]
                                        (map #(arg/set-arg-node-argument
                                               % :output (get node :id) (get parameter :key))
                                             loss-term-seq))
                                      trainable-parameters))]
    (concat node-losses parameter-losses)))


(defn- generate-loss-function
  [network nodes output-bindings loss]
  (let [network (remove-existing-loss-terms network)
        node-losses (->> nodes
                         (mapcat (partial generate-node-loss-terms network)))
        output-losses (->> output-bindings
                           (filter #(and (get % :loss)
                                         (get % :stream)))
                           (map (fn [{:keys [node-id stream loss]}]
                                  (-> loss
                                      (arg/set-arg-node-output :output node-id)
                                      (arg/set-arg-stream :labels stream)))))]
    (concat loss output-losses node-losses)))


(defn- merge-streams
  [stream-map graph]
  (reduce (fn [graph [stream size]]
            (if-not (contains? (get graph :streams) stream)
              (graph/add-stream graph stream (graph/create-stream-descriptor
                                              (long size)))
              graph))
          graph
          stream-map))


(defn- set-loss-terms
  [loss-term-vec graph]
  ;;map each loss term to the node it is most associated with (its output)
  ;;and attach it to that node in the graph.  Generate parameters and then
  ;;create a new vector of loss terms with the added information.
  (reduce (fn [[graph loss-term-vec] loss-term]
            (when-not (contains? (-> (graph/get-node-metadata loss-term)
                                     :passes
                                     set)
                                 :loss)
              (throw (ex-info "Loss term does not contain the loss pass in it's metadata"
                              {:loss-term loss-term
                               :metadata (graph/get-node-metadata loss-term)})))
            (let [node-id (->> (graph/get-node-arguments loss-term)
                               (map :node-id)
                               (remove nil?)
                               first)
                  _ (when-not node-id
                      (throw (ex-info "failed to find node for loss term"
                                      {:term loss-term
                                       :arguments (vec (graph/get-node-arguments
                                                        loss-term))})))
                  [graph term-id] (graph/add-node graph loss-term [node-id])]
              [graph (conj loss-term-vec (assoc loss-term :id term-id))]))
          [graph []]
          loss-term-vec))


(defn- generate-loss-term-parameters
  "Generating loss term parameters modifies the nodes associated with those
parameters by adding buffer-ids in some cases."
  [network stream-map loss-term-vec]
  (let [[graph loss-term-vec] (->> (get network :layer-graph)
                                   (merge-streams stream-map)
                                   (set-loss-terms loss-term-vec))
        graph (graph/generate-parameters graph)]
    [(assoc network :layer-graph graph)
     (->> loss-term-vec
          (map :id)
          (mapv #(graph/get-node graph %)))]))


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

You can specify a loss here directly or you can specify loss terms around the graph.
Any terms in the graph are coalesced and appended to the passed-in loss to build a
datastructure describing the final loss function.
{:buffers map id->{:size}
 :forward where incoming/outgoing maps to buffer id
 :backward where incoming/outgoing maps to buffer id}"
  [network stream-map
   & {:keys [optimizer keep-non-trainable? loss-fn]
      :or {optimizer (adam/adam)
           loss-function []}}]
  (check-for-io-bindings network)
  (let [forward-traversal (->> (create-forward-traversal network)
                               (filter-traversal network :training))
        [forward-with-buffers buffer-map] (traversal->buffers forward-traversal {})
        backward-pass (if keep-non-trainable?
                        forward-traversal
                        (remove-non-trainable network forward-traversal))
        forward-traversal-nodes (->> backward-pass
                                     reverse
                                     (map :id)
                                     (map #(network/network->node network %)))
        ;;If the loss function is regenerated then any loss term parameters are lost.
        [network loss-fn]
        (if-not (get-in network [:traversal :loss-function])
          (->> (generate-loss-function network
                                       forward-traversal-nodes
                                       (get-output-bindings network)
                                       loss-fn)
               (generate-loss-term-parameters network stream-map))
          [network (get-in network [:traversal :loss-function])])]
    (update network
            :traversal
            #(merge %
                    {:forward (-> forward-with-buffers
                                  clean-traversal-incoming-outgoing)
                     :backward (-> backward-pass
                                   (traversal->buffers buffer-map)
                                   first
                                   reverse-forward-traversal
                                   clean-traversal-incoming-outgoing)
                     :buffers (forward-traversal->buffer-map network forward-with-buffers)
                     :type :training
                     :stream-map (->> (get-in network [:layer-graph :streams])
                                      (map (fn [[k v]]
                                             [k (graph/stream->size
                                                 (get network :layer-graph) k)]))
                                      (into {}))
                     :optimizer optimizer
                     :loss-function loss-fn}))))


(defn network->inference-traversal
  "Similar to network->gradient-descent however in this case we have the option
  of optimising for memory which means we can aggressively reuse buffers *or*
  optimising for speed in which case the result is the forward pass of gradient descent
  and we expect implementations to have multiple batches in flight simultaneously.  We
  default to optimising for memory because this avoids OOM situations with large networks."
  [{:keys [layer-graph] :as network} stream-map]
  (check-for-io-bindings network)
  (let [network (remove-existing-loss-terms network)
        forward-traversal (->> (create-forward-traversal network)
                               (filter-traversal network :inference)
                               (#(traversal->buffers % {}))
                               first)]
    (update network
            :traversal
            #(merge
              %
              {:forward (clean-traversal-incoming-outgoing forward-traversal)
               :buffers (forward-traversal->buffer-map network forward-traversal)
               :type :inference
               :stream-map stream-map}))))


(defn- traversal->buffer-set
  [traversal]
  (->> traversal
       (mapcat (fn [{:keys [incoming outgoing]}]
                 (concat incoming outgoing)))
       set))


(defn network->forward-buffer-set
  "Get the set of buffers used for the forward pass"
  [network]
  (->> (get-in network [:traversal :forward])
       traversal->buffer-set))


(defn network->backward-buffer-set
  "Get the set of buffers used for the backward pass"
  [network]
  (->> (get-in network [:traversal :backward])
       traversal->buffer-set))
