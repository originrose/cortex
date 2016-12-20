(ns cortex.nn.traverse
  "Various graph traversal algorithms needed in order to implement
either inference or gradient descent on a layer graph."
  (:require [cortex.nn.build :as build]
            [cortex.nn.layers :as layers]
            [clojure.set :as c-set]))


(defn- check-node-id
  [built-network node-id]
  (when-not (get-in built-network [:layer-graph :id->node-map node-id])
    (throw (ex-info "Failed to find node id in graph"
                    {:node-id node-id
                     :graph-nodes (keys (get-in built-network [:layer-graph :id->node-map]))}))))


(defn bind-input
  "Bind a specific node to a dataset stream.  Returns a new network."
  [built-network node-id dataset-stream-name]
  ;;Multiple a node can bind to only one stream but one stream may be bound
  ;;to multiple nodes.
  (check-node-id built-network node-id)
  (assoc-in built-network [:input-bindings node-id] {:stream dataset-stream-name}))


(defn bind-output-infer
  "Enable output from a given graph node.  Since this is inference we do not need
to worry about loss functions."
  [built-network node-id output-name]
  (check-node-id built-network node-id)
  ;;Any node can produce an output stream but they have to be uniquely named.  A node
  ;;can bind to multiple output names.
  (assoc-in built-network [:output-bindings output-name] {:node-id node-id}))


(defn bind-output-train
  "Enable output and bind it to a dataset stream for training.  If a loss function isn't specified
then one will be chosen automatically based on the layer type."
  [built-network node-id dataset-stream-name & {:keys [loss-function]}]
  (check-node-id built-network node-id)
  (let [loss-function (if loss-function
                        loss-function
                        (layers/auto-bind-loss (get-in built-network
                                                       [:layer-graph :id->node-map node-id])))]
    (assoc-in built-network [:output-bindings dataset-stream-name] {:node-id node-id
                                                                    :loss loss-function})))


(defn get-dataset-bindings
  "get a sequence of maps of:
{:node-id
 :dataset-stream
 :direction [:input :output]
 :loss-function (if output)}."
  [{:keys [input-bindings output-bindings]}]
  (concat (map (fn [[node-id {:keys [stream]}]]
                 {:node-id node-id
                  :dataset-stream stream
                  :direction :input})
               input-bindings)
          (map (fn [[dataset-stream {:keys [node-id loss-function]}]]
                 {:node-id node-id
                  :dataset-stream dataset-stream
                  :direction :output
                  :loss-function loss-function}))))


(defn- create-forward-traversal
  "A forward traversal is a linear dfs order sequence.
There is an optional argument to remove nodes of a particular type from
the traversal.

Each item in the sequence is a map of:
{:incoming ()
 :id
 :outgoing ()
}"
  [{:keys [layer-graph input-bindings output-bindings] :as built-network} remove-type-set]
  (let [{:keys [id->node-map edges]} layer-graph
        ;;Removing all nodes of a given type from the network if they aren't marked as input or output.
        keep-node-set (set (concat (keys input-bindings)
                                   (map :node-id (vals output-bindings))))

        ;;Ensure the remove set does not contain any of the keep node set.
        remove-id-set (c-set/difference
                       (->> (filter #(contains? remove-type-set
                                                (get % :type))
                                    (vals id->node-map))
                            (map :id)
                            set)
                       keep-node-set)
        ;;Trim items by type and patch up edges.
        edges (if (empty? remove-id-set)
                edges
                (let [alias-map (->> (map (fn [[top bottom]]
                                            (cond
                                              (and (contains? remove-id-set top)
                                                   (contains? remove-id-set bottom))
                                              nil
                                              (contains? remove-id-set top)
                                              [top bottom]
                                              (contains? remove-id-set bottom)
                                              [bottom top]
                                              :else
                                              nil))
                                          edges)
                                     (into {}))]
                  (->> (map (fn [[top bottom]]
                              (if (contains? remove-id-set top)
                                nil
                                [(get alias-map top top)
                                 (get alias-map bottom bottom)]))
                            edges)
                       (remove nil?))))
        ;;walk over edges and create master keep node set of all connected edges
        ;;Iterative algorithm with runtime at most of N where N is number of edges.
        keep-node-set (loop [keep-node-set keep-node-set]
                        (let [new-keep-node-set
                              (reduce (fn [keep-node-set [top bottom]]
                                        (if (contains? keep-node-set top)
                                          (conj keep-node-set bottom)
                                          keep-node-set))
                                      keep-node-set
                                      edges)]
                          (if-not (= keep-node-set new-keep-node-set)
                            (recur new-keep-node-set)
                            new-keep-node-set)))

        ;;Remove all edges that do not participate in the keep node set.
        edges (filter #(and (contains? keep-node-set (first %))
                            (contains? keep-node-set (second %)))
                      edges)
        [roots leaves] (build/edges->roots-and-leaves edges)
        parent->child-map (build/edges->parent->child-map edges)
        child->parent-map (build/edges->child->parent-map edges)]

    (->> (build/edges->dfs-seq edges :roots parent->child-map)
         (drop 1)
         (map (fn [id]
                {:incoming (get child->parent-map id)
                 :id id
                 :outgoing (get parent->child-map id)})))))



(defn- reverse-forward-traversal
  "See create-forward-traversal.  Reverse of same sequence."
  [forward-traversal]
  (->> forward-traversal
       reverse
       (map (fn [{:keys [incoming outgoing] :as traverse-item}]
              (assoc traverse-item
                     :incoming outgoing
                     :outgoing incoming)))))


(defn- forward-traversal->gd-buffers
  [traverse-item traversal buffer-map input-count output-count id->node-map]
  (when traverse-item
    (let [{:keys [incoming id outgoing]} traverse-item
          size (get-in id->node-map
                              [id :output-size])
          [input-idx input-count] (if (empty? incoming)
                                    [input-count (inc input-count)]
                                    [nil input-count])
          [output-idx output-count] (if (empty? outgoing)
                                      [output-count (inc output-count)]
                                      [nil output-count])
          new-buffer (cond-> {:size size}
                       output-idx
                       (assoc :output {output-idx size}))
          [input-buffer incoming-id] (if input-idx
                                       (let [input-size (get-in id->node-map [id :input-size])]
                                        [{:size input-size
                                          :inputs {input-idx input-size}}
                                         (keyword (str (name id) "-input-" input-idx))])
                                       [nil nil])
          buffer-map (cond-> (assoc buffer-map id new-buffer)
                       input-buffer
                       (assoc incoming-id input-buffer))
          incoming (if incoming-id
                     [{:id incoming-id
                       :input-idx input-idx}]
                     incoming)]
      (cons
       [{:incoming incoming :id id :outgoing id} buffer-map]
       (lazy-seq (forward-traversal->gd-buffers
                  (first traversal) (rest traversal) buffer-map
                  input-count output-count id->node-map))))))


(defn network->gradient-descent
  "Given network create master buffer list,
two traversals (forward,backward)
and input and output buffer lists.
Each traversal is sequence of maps like in create-forward-traversal
exception the incoming and outgoing ids are buffer ids.
{:buffers map id->{:size}
 :forward where incoming/outgoing maps to buffer id
 :backward where incoming/outgoing maps to buffer id}"
  [built-network & {:keys [remove-type-set optimiser]
                    :or {remove-type-set #{:input}
                         optimiser (layers/adam)}}]
  (let [forward-traversal (create-forward-traversal built-network remove-type-set)
        {:keys [id->node-map]} (get built-network :layer-graph)
        id->outgoing #(map (fn [{:keys [id] :as item}]
                             (assoc item :outgoing [id]))
                           %)
        forward-traversal-data (forward-traversal->gd-buffers
                                (first forward-traversal) (rest forward-traversal) {}
                                0 0 id->node-map)
        forward-traversal (map first forward-traversal-data)
        buffer-map (second (last forward-traversal-data))]
    {:forward forward-traversal
     :buffers buffer-map
     :backward (reverse-forward-traversal forward-traversal)
     :type :gradient-descent
     :optimiser optimiser}))


(defn- allocate-buffer
  "Assumption is that the free buffers are sorted by size"
  [id output-size free-buffers]
  (let [free-buffers (sort-by :size free-buffers)
        compare-condition #(> (get % :size) output-size)
        next-buffer (first (filter compare-condition
                                   free-buffers))]
    (if next-buffer
      [next-buffer (remove compare-condition free-buffers)]
      (let [next-buffer (or (last free-buffers)
                            {:id id})]
        [(assoc next-buffer :size output-size) (drop-last free-buffers)]))))


(defn- free-in-flight-buffers
  "Free any in-flight-buffers with zero refcount.
returns [free-buffers in-flight-buffers]"
  [incoming in-flight-buffers]
  (reduce (fn [[free-buffers in-flight-buffers] incoming-id]
            (let [in-flight-buffer (get in-flight-buffers incoming-id)
                  ref-count (dec (get in-flight-buffer :ref-count))]
              (if (= 0 ref-count)
                [(conj free-buffers in-flight-buffer) (dissoc in-flight-buffers incoming-id)]
                [free-buffers (assoc in-flight-buffers incoming-id in-flight-buffer)])))
          [[] in-flight-buffers]
          incoming))


(defn- forward-traversal->inference
  "Given a basic forward traversal generate a memory-optimised forward pass
that uses the reasonably small buffers."
  [traverse-item forward-traverse-seq all-buffers
   free-buffers in-flight-buffers id->node-map
   input-count output-count]
  (when traverse-item
    (let [{:keys [incoming id outgoing]} traverse-item
          [input-idx input-count] (if (empty? incoming)
                                    [input-count (inc input-count)]
                                    [nil input-count])
          [output-idx output-count] (if (empty? outgoing)
                                      [output-count (inc output-count)]
                                      [nil output-count])
          incoming-id (if input-idx
                        (keyword (str (name id) "-input-" input-idx))
                        nil)
          ;;allocate input buffer if we are top of chain
          input-size (get-in id->node-map [id :input-size])
          [input-buffer free-buffers] (if input-idx
                                        (allocate-buffer incoming-id
                                                         input-size
                                                         free-buffers)
                                        [nil free-buffers])

          input-buffer (when input-buffer
                         (update input-buffer :inputs #(assoc % input-idx input-size)))

          in-flight-buffers (if input-idx
                              (assoc in-flight-buffers
                                     incoming-id (assoc input-buffer
                                                        :ref-count 1))
                              in-flight-buffers)
          ;;Allocate input buffer
          [next-buffer free-buffers] (allocate-buffer id
                                                      (get-in id->node-map
                                                              [id :output-size])
                                                      free-buffers)
          ;;mark in flight
          in-flight-buffers (assoc in-flight-buffers
                                   id (assoc next-buffer
                                             :ref-count (count outgoing)))

          ;;mangle incoming so it points to buffer ids, not to node ids
          new-incoming (if input-idx
                         [{:id incoming-id :input-idx input-idx}]
                         (map #(get-in in-flight-buffers [% :id]) incoming))
          [next-free in-flight-buffers] (free-in-flight-buffers (if input-idx
                                                                  [incoming-id]
                                                                  incoming)
                                                                in-flight-buffers)
          ;;Keep track of all buffers
          all-buffers (cond-> (assoc all-buffers
                                     (get next-buffer :id)
                                     (dissoc next-buffer :ref-count))
                        incoming-id
                        (assoc (get input-buffer :id) (dissoc input-buffer
                                                              :ref-count)))]
      (cons
       [{:incoming new-incoming :id id :outgoing (get next-buffer :id)} all-buffers]
       (lazy-seq (forward-traversal->inference (first forward-traverse-seq)
                                               (rest forward-traverse-seq)
                                               all-buffers (concat free-buffers
                                                                   next-free)
                                               in-flight-buffers id->node-map
                                               input-count output-count))))))


(defn network->inference
  "Similar to network->gradient-descent however in this case we have the option
  of optimising for memory which means we can aggressively reuse buffers *or*
  optimising for speed in which case the result is the forward pass of gradient descent
  and we expect implementations to have multiple batches in flight simultaneously.  We
  default to optimising for memory because this avoids OOM situations with large networks."
  [{:keys [layer-graph] :as built-network} & {:keys [optimise-type remove-type-set]
                                              :or {optimise-type :memory
                                                   remove-type-set #{:dropout :input}}}]

  (if (= optimise-type :memory)
    (let [basic-forward (create-forward-traversal built-network remove-type-set)
          forward-traverse-seq (forward-traversal->inference (first basic-forward)
                                                             (rest basic-forward)
                                                             {}
                                                             []
                                                             {}
                                                             (get layer-graph
                                                                  :id->node-map)
                                                             0 0)]
      {:forward (map first forward-traverse-seq)
       :buffers (second (last forward-traverse-seq))
       :type :inference})
    (-> (network->gradient-descent built-network :remove-type-set remove-type-set)
        (dissoc :backward)
        (assoc :type :inference))))
