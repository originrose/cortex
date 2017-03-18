(ns cortex.nn.traverse
  "Various graph traversal algorithms needed in order to implement
  either inference or gradient descent on a layer graph.

  Note that input-bindings are maps from node-id to stream
  while output bindings are maps from node-id to {:stream :loss}."
  (:require
    [clojure.set :as c-set]
    [clojure.core.matrix :as m]
    [cortex.loss :as loss]
    [cortex.argument :as arg]
    [cortex.graph :as graph]
    [cortex.optimize :as optimize]
    [cortex.optimize.adam :as adam]
    [cortex.nn.layers :as layers]))



(defn forward-traversal
  "A forward traversal is a linear dfs order sequence.
  There is an optional argument to remove nodes of a particular type from
  the traversal.

  Each item in the sequence is a map of:
  {:incoming buffer-map-seq
  :id
  :outgoing buffer-map-seq}
  "
  [{:keys [compute-graph] :as network}]
  (let [{:keys [input-bindings output-bindings]} (get network :traversal)
        ;;Remove all edges that do not participate in the keep node set.
        child->parent-map (graph/child->parent-map compute-graph)
        output-bindings (->> output-bindings
                             (map (fn [[k v]]
                                    [k (dissoc v :stream)]))
                             (into {}))
        nodes-depth-first (graph/dfs-seq compute-graph)]
    (->> nodes-depth-first
        (reduce
          (fn [[retval id->buffer-map] id]
            (let [node (graph/get-node compute-graph id)
                  output-dims (graph/node->output-dimensions node)
                  output-buffers (if (= 1 (count output-dims))
                                   (-> (assoc
                                         (if-let [output-binding (get output-bindings id)]
                                           (merge {:output-id id}
                                                  output-binding)
                                           {:id id})
                                         :dimension (graph/node->output-dimension node))
                                       vector)
                                   (->> output-dims
                                        (map-indexed (fn [idx output-dim]
                                                       {:id (keyword (str (name id)
                                                                          "-"
                                                                          (+ idx 1)))
                                                        :dimension output-dim}))
                                        vec))
                  ;;Take the input bindings and the incoming ids and ensure that all buffers
                  ;;have the correct id and have dimensions on them.
                  incoming (concat
                             (->> [id]
                                  (map input-bindings)
                                  (remove nil?)
                                  (map #(assoc %
                                               :dimension
                                               (graph/node->input-dimension node))))
                             (->> (get child->parent-map id)
                                  ;;Find the parent output dimension that targets this node.
                                  ;;This accounts for the possibility that a parent could have
                                  ;;different sized outputs for different children.
                                  (map (fn [parent-id]
                                         (let [output-dims (get id->buffer-map parent-id)
                                               retval (if (= 1 (count output-dims))
                                                        (first output-dims)
                                                        (first (filter
                                                                 #(= id (get-in %
                                                                                [:dimension :id]))
                                                                 output-dims)))]
                                           (when-not retval
                                             (throw (ex-info "Failed to find input buffer"
                                                             {:node node
                                                              :parent parent-id
                                                              :parent-output-dims output-dims})))
                                           retval)))))]
              [(conj retval {:incoming (vec incoming)
                             :id id
                             :outgoing output-buffers})
               (assoc id->buffer-map id output-buffers)]))
          [[] {}])
        first)))


(defn filter-traversal
  "Removes bits of the traversal that aren't needed (e.g. no dropout used in
  inference), and then corrects the input/output ids accordingly."
  [{:keys [compute-graph] :as network} pass-type traversal]
  (->> traversal
       (reduce (fn [[traversal input-alias-map] {:keys [incoming id] :as entry}]
                 (let [graph-node (graph/get-node compute-graph id)
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
                 (let [graph-node (get-in compute-graph [:nodes id])
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

(defn- buffer-desc->map-key
  [buffer-desc]
  (select-keys buffer-desc [:id :stream :output-id]))


(defn traversal->buffers
  "Traversals initial hold id of incoming nodes.  For the next steps
we need the incoming and outgoing edges to hold unique ids such that
the incoming buffer of the next step points to the outgoing buffer of
the previous step."
  [traversal buffer-map]
  (->> traversal
       (mapcat #(concat (get % :incoming)
                        (get % :outgoing)))
       (concat (vals buffer-map))
       (group-by buffer-desc->map-key)
       (map (fn [[buf-key buf-val-seq]]
              (let [val-map (group-by #(graph/dimensions->size (get % :dimension))
                                      buf-val-seq)]
                (when-not (= 1 (count val-map))
                  (throw (ex-info "Multiple sized buffers detected for key"
                                  {:buffer-key buf-key
                                   :buffer-values buf-val-seq})))
                [buf-key (first buf-val-seq)])))
       (into {})))


(defn- reverse-forward-traversal
  "See create-forward-traversal.  Reverse of same sequence."
  [forward-traversal]
  (->> forward-traversal
       reverse
       (map (fn [{:keys [incoming outgoing] :as traverse-item}]
              (assoc traverse-item
                     :incoming outgoing
                     :outgoing incoming)))))


(defn- clean-buffer-map
  "Get just the description info for a buffer description map."
  [buffer-desc]
  (select-keys buffer-desc [:id :stream :output-id]))


(defn- clean-traversal-incoming-outgoing
  "Make the incoming and outgoing edges actually valid buffer keys
which means removing extra information from them."
  [traversal]
  (map (fn [entry]
         (-> entry
             (update :incoming #(map clean-buffer-map %))
             (update :outgoing #(map clean-buffer-map %))))
       traversal))


(defn- remove-non-trainable
  [network traversal]
  (-> (reduce (fn [[keep-set traversal] {:keys [incoming id] :as item}]
                (let [keep? (or (seq (filter #(contains? keep-set (get % :id)) incoming))
                                (graph/any-trainable-arguments? (graph/get-node
                                                                 (get  network :compute-graph)
                                                                  id)))]
                  (if keep?
                    [(conj keep-set id) (conj traversal item)]
                    [keep-set traversal])))
              [#{} []]
              traversal)
      second))


(defn add-training-traversal
  "Given a network create master buffer list, traversals (forward,backward)
  and input and output buffer lists.

  Each traversal is a sequence of maps like in create-forward-traversal
  except the incoming and outgoing ids are buffer ids.  Input bindings
  are pairs of node to stream name.  Output bindings for gradient descent
  are also pairs of node-id to stream name or they can be
  pairs of node-id to [stream-name loss-function].

  You can specify a loss here directly or you can specify loss terms around the graph.
  Any terms in the graph are coalesced and appended to the passed-in loss to build a
  datastructure describing the final loss function.

  Note that input-bindings are maps from stream to node-id
  while output-bindings are maps from node-id to {:stream :loss}!

  {:buffers <id->{:size}>
   :forward <where incoming/outgoing maps to buffer id>
   :backward <where incoming/outgoing maps to buffer id>}
  "
  [network stream-map
   & {:keys [optimizer keep-non-trainable? loss-fn]
      :or {optimizer (adam/adam)
           loss-function []}}]
  (let [forward-traversal (forward-traversal network)
        forward-traversal (filter-traversal network :training forward-traversal)
        buffer-map (traversal->buffers forward-traversal {})
        backward-pass (if keep-non-trainable?
                        forward-traversal
                        (remove-non-trainable network forward-traversal))
        forward-traversal-nodes  (->> backward-pass
                                      reverse
                                      (map :id)
                                      (map #(graph/get-node (:compute-graph network) %)))]
    (update network
            :traversal
            #(merge %
                    {:forward (-> forward-traversal
                                  clean-traversal-incoming-outgoing)
                     :backward (-> backward-pass
                                   reverse-forward-traversal
                                   clean-traversal-incoming-outgoing)
                     :buffers buffer-map
                     :type :training}))))


(defn add-forward-traversal
  "Similar to network->gradient-descent however in this case we have the option
  of optimizing for memory which means we can aggressively reuse buffers *or*
  optimising for speed in which case the result is the forward pass of gradient descent
  and we expect implementations to have multiple batches in flight simultaneously.  We
  default to optimising for memory because this avoids OOM situations with large networks."
  [{:keys [compute-graph] :as network} stream-map]
  (let [forward-traversal (->> (forward-traversal network)
                               (filter-traversal network :inference))]
    (update network
            :traversal
            #(merge
              %
              {:forward (clean-traversal-incoming-outgoing forward-traversal)
               :buffers (traversal->buffers forward-traversal {})
               :type :inference
               :stream-map stream-map}))))


(defn- traversal-buffers
  [traversal]
  (->> traversal
       (mapcat (fn [{:keys [incoming outgoing]}]
                 (concat incoming outgoing)))
       set))


(defn get-forward-buffers
  "Get the set of buffers used for the forward pass"
  [network]
  (->> (get-in network [:traversal :forward])
       traversal-buffers))


(defn get-backward-buffers
  "Get the set of buffers used for the backward pass"
  [network]
  (->> (get-in network [:traversal :backward])
       traversal-buffers))
