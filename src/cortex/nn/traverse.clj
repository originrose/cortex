(ns cortex.nn.traverse
  "Various graph traversal algorithms needed in order to implement
  either inference or gradient descent on a layer graph.

  Note that input-bindings are maps from node-id to stream
  while output bindings are maps from node-id to {:stream :loss}."
  (:require [clojure.set :as c-set]
            [clojure.core.matrix :as m]
            [cortex.argument :as arg]
            [cortex.graph :as graph]
            [cortex.optimize :as optimize]
            [cortex.optimize.adam :as adam]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
            [cortex.loss.core :as loss]
            [cortex.util :as util]))


(defn- ensure-unique-buffer-id
  [{:keys [id] :as buffer} buffer-ids]
  (if (contains? buffer-ids id)
    (assoc buffer :id
           (util/generate-id (name id) buffer-ids))
    buffer))


(defn forward-traversal
  "A forward traversal is a linear dfs order sequence.
  There is an optional argument to remove nodes of a particular type from
  the traversal.

  Each item in the sequence is a map of:
  {:incoming buffer-map-seq
  :id
  :outgoing buffer-map-seq}
  "
  [network]
  (let [;;For now we compute the traversal ignorant of the loss terms
        compute-graph (graph/filter-graph (network/network->graph network)
                                          network/is-non-loss-node?)
        ;;Remove all edges that do not participate in the keep node set.
        child->parent-map (graph/child->parent-map compute-graph)
        nodes-depth-first (graph/dfs-seq compute-graph)]
    (->> nodes-depth-first
        (reduce
         (fn [[retval id->buffer-map buffer-ids] id]
            (let [node (graph/get-node compute-graph id)
                  output-dims (graph/node->output-dimensions node)
                  [output-buffers buffer-ids]
                  (->> (if (= 1 (count output-dims))
                         [{:id id
                           :dimension (graph/node->output-dimension node)}]
                         (->> output-dims
                              (map-indexed (fn [idx output-dim]
                                             {:id (keyword (str (name id)
                                                                "-"
                                                                (+ idx 1)))
                                              :dimension output-dim}))))
                       ;;tranlate the :id :dimension map possibly into :stream :dimensions
                       ;;if the output dimension coming from the node indicates that it is
                       ;;simply the incoming stream.
                       (map (fn [{:keys [id dimension] :as buf}]
                              (if (contains? dimension :stream)
                                {:stream (get dimension :stream)
                                 :dimension (dissoc dimension :stream)}
                                buf)))
                       ;;Ensure that the output buffers do in fact have unique ids.  At this point
                       ;;Items with multiple output buffers will buffers where the dimension member
                       ;;has an id entry that points to the child.  Thus the child filters its possible set
                       ;;of input buffers when there are several possible to remove input buffers that
                       ;;do not explicitly point to this child.
                       (reduce (fn [[output-buffers buffer-ids] buffer]
                                 (if-not (contains? buffer :id)
                                   [(conj output-buffers buffer) buffer-ids]
                                   (let [updated-buffer (ensure-unique-buffer-id
                                                         buffer buffer-ids)]
                                     [(conj output-buffers updated-buffer)
                                      (conj buffer-ids (get updated-buffer :id))])))
                               [[] buffer-ids]))
                  incoming (->> (get child->parent-map id)
                                ;;Find the parent output dimension that targets this node.  This
                                ;;accounts for the possibility that a parent could have
                                ;;different sized outputs for different children.  This means
                                ;;that nodes that produce many buffers must identify which child
                                ;;gets which buffer.  They do this by embedding an id on the
                                ;;output dimension which indicates the intended child.
                                (map (fn [parent-id]
                                       (let [output-dims (get id->buffer-map parent-id)
                                             retval (if (= 1 (count output-dims))
                                                      (let [output-dim (first output-dims)]
                                                        (if (contains? output-dim :id)
                                                          output-dim
                                                          (assoc output-dim :id parent-id)))
                                                      (first (filter
                                                              #(= id (get-in %
                                                                             [:dimension :id]))
                                                              output-dims)))]
                                         (when-not retval
                                           (throw (ex-info "Failed to find input buffer"
                                                           {:node node
                                                            :parent parent-id
                                                            :parent-output-dims output-dims})))
                                         ;;It is very confusing but right here in the code
                                         ;;retval is the single buffer meant for this child from
                                         ;;this parent.  It should be only identified by the
                                         ;;parent id but not any other information If
                                         ;;information is leaked (like stream or id) then it can
                                         ;;become difficult later to discern where a buffer came
                                         ;;from.
                                         (-> retval
                                             (update :dimension graph/clear-dimension-identifiers)
                                             (dissoc :stream))))))]
              [(conj retval {:incoming (vec incoming)
                             :id id
                             :outgoing output-buffers})
               (assoc id->buffer-map id output-buffers)
               buffer-ids]))
          [[] {} #{}])
        first)))

(defn- graph-node->pass-type
  [graph-node pass-type]
  (if (get graph-node :non-trainable?)
    :inference
    pass-type))


(defn filter-traversal
  "Removes bits of the traversal that aren't needed (e.g. no dropout used in
  inference), and then corrects the input/output ids accordingly."
  [{:keys [compute-graph] :as network} pass-type traversal]
  (->> traversal
       ;;Logically if a node is removed here then that means that it assigns its
       ;;input to its output.
       (reduce (fn [[traversal input-alias-map] {:keys [incoming id outgoing] :as entry}]
                 (let [graph-node (graph/get-node compute-graph id)
                       pass-type (graph-node->pass-type graph-node pass-type)
                       pass-set (layers/get-pass-set graph-node)
                       new-incoming (flatten (map #(get input-alias-map (get % :id) %)
                                                  incoming))]
                   (if (contains? pass-set pass-type)
                     [(conj traversal
                            (assoc entry
                                   :incoming new-incoming))
                      input-alias-map]
                     [(conj traversal entry)
                      (reduce (fn [input-alias-map {:keys [id stream]}]
                                (assoc input-alias-map
                                       (or id stream)
                                       (concat new-incoming
                                               (->> outgoing
                                                    (filter #(contains? % :stream))))))
                              input-alias-map
                              outgoing)])))
               [[] {}])
       first
       reverse
       (reduce (fn [[traversal output-alias-map] {:keys [id outgoing] :as entry}]
                 (let [graph-node (get-in compute-graph [:nodes id])
                       pass-type (graph-node->pass-type graph-node pass-type)
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
  "Traversals initial hold id of incoming nodes.  For the next steps we need the incoming and
  outgoing edges to hold unique ids such that the incoming buffer of the next step points to the
  outgoing buffer of the previous step."
  [traversal]
  (->> traversal
       (mapcat #(concat (get % :incoming)
                        (get % :outgoing)))
       (group-by buffer-desc->map-key)
       (map (fn [[buf-key buf-val-seq]]
              (let [val-map (group-by #(graph/dimensions->size (get % :dimension))
                                      buf-val-seq)]
                (when-not (= 1 (count val-map))
                  (throw (ex-info "Multiple sized buffers detected for key"
                                  {:buffer-key buf-key
                                   :buffer-values buf-val-seq})))
                [buf-key (dissoc (first buf-val-seq) :id :stream)])))
       (into {})))


(defn stream-arguments->buffers
  [network buffer-map graph-type]
  (->> (concat (network/graph-streams network graph-type)
               (network/augmented-streams network graph-type))
       (map (fn [[k dim]]
              [{:stream k} {:dimension dim}]))
       (into {})
       (merge buffer-map)))


(defn- reverse-forward-traversal
  "See create-forward-traversal.  Reverse of same sequence."
  [forward-traversal]
  (->> forward-traversal
       reverse
       ;;Force computation here so that errors are caught with some semblence of a reasonable
       ;;stack trace
       (mapv (fn [{:keys [incoming outgoing] :as traverse-item}]
               (assoc traverse-item
                      :incoming outgoing
                      :outgoing incoming)))))


(defn remove-non-trainable
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

(defn- clean-traversal
  "Remove extraneous information from the io buffer list of the nodes to keep the traversal
datastructure as precise as possible.  This will remove at least the dimension member which from
the client's perspective is located in the buffers section."
  [trav]
  (let [cleaner-fn (fn [io-seq]
                     (mapv (fn [io-entry]
                             (select-keys io-entry [:id :stream]))
                           io-seq))]
    (mapv #(-> %
               (update :incoming cleaner-fn)
               (update :outgoing cleaner-fn))
          trav)))

(defn- network->forward-traversal-and-buffers
  "Given a network return a tuple or the forward traversal and the buffer allocation"
  [network graph-type]
  (let [pre-buffer-forward (->> (forward-traversal network)
                                (filter-traversal network graph-type))]
    [(clean-traversal pre-buffer-forward)
     (traversal->buffers pre-buffer-forward)]))


(defn training-traversal
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
  [network & {:keys [keep-non-trainable?]}]
  (let [[forward-traversal buffers] (network->forward-traversal-and-buffers
                                     network :training)
        backward-traversal (-> (if keep-non-trainable?
                                 forward-traversal
                                 (remove-non-trainable network forward-traversal))
                               reverse-forward-traversal)
        backward-id-set (set (map :id backward-traversal))
        forward-traversal (map (fn [{:keys [id] :as entry}]
                                 (cond-> entry
                                   (and (not (contains? backward-id-set id))
                                        (get-in network [:compute-graph :nodes id :non-trainable?]))
                                   (assoc :pass :inference)))
                               forward-traversal)]
    ;;Layers not involved in the backward traversal should have their passes set to inference.

    {:forward forward-traversal
     :backward backward-traversal
     :buffers (stream-arguments->buffers network buffers :training)
     :type :training}))


(defn- trainable-loss-term-argument?
  [network nodes-in-traversal loss-term]
  ;;Node outputs or parameters are trainable
  (when (and (condp = (get loss-term :type)
               :node-output true
               :node-argument true
               false)
             ;;These loss term arguments are capable of producing gradients
             (get loss-term :gradients?)
             (contains? nodes-in-traversal
                        (get loss-term :node-id)))
    (if (= (get loss-term :type) :node-output)
      ;;If node output then we are finished
      true
      ;;If node parameter, however, we need to check if the parameter is trainable.
      (let [{:keys [node-id argument]} loss-term
            net-graph (network/network->graph network)
            node (graph/get-node net-graph node-id)
            node-arg (graph/get-node-argument node argument)]
        (get node-arg :gradients?)))))


(defn gradient-loss-function
  "Filter the general network loss function terms to remove terms that do not contribute
  to training gradients.  Loss functions that do not produce any gradients
  *used in training* are filtered out.  The determination of used in training means
  that the node takes part of the backward pass and that some of the parameters that the loss term
  can produce gradients for are actually trainable or that it produces node output gradients."
  [network {:keys [backward] :as traversal}]
  (when-not backward
    (throw (ex-info "Traversal does not appear to have a backward pass."
                    {:traversal traversal})))
  (let [pure-loss-fn (network/loss-function network)
        nodes-in-traversal (->> backward
                                (map :id)
                                set)]
    (->> pure-loss-fn
         (filter (fn [loss-term]
                   ;;Are any of the arguments to the loss term node outputs or
                   ;;hooked to trainable parameters?
                   (seq (->> (graph/get-node-arguments loss-term)
                             (filter (partial trainable-loss-term-argument?
                                              network
                                              nodes-in-traversal))))))
         set)))


(defn inference-traversal
  "Similar to network->gradient-descent however in this case we have the option
  of optimizing for memory which means we can aggressively reuse buffers *or*
  optimising for speed in which case the result is the forward pass of gradient descent
  and we expect implementations to have multiple batches in flight simultaneously.  We
  default to optimising for memory because this avoids OOM situations with large networks."
  [network]
  (let [[forward-traversal buffers] (network->forward-traversal-and-buffers
                                     network :inference)]
    {:forward forward-traversal
     :buffers (stream-arguments->buffers network buffers :inference)
     :type :inference}))


(defn- traversal-buffers
  [traversal]
  (->> traversal
       (mapcat (fn [{:keys [incoming outgoing]}]
                 (concat incoming outgoing)))
       set))


(defn get-forward-buffers
  "Get the set of buffers used for the forward pass"
  [traversal]
  (->> (get traversal :forward)
       traversal-buffers))


(defn get-backward-buffers
  "Get the set of buffers used for the backward pass"
  [traversal]
  (->> (get traversal :backward)
       traversal-buffers))


(defn record-first-seen-buffer-id-index
  "Return a map of buffer id to the first time we have seen it"
  [buffer-list]
  (->> buffer-list
       (reduce (fn [retval {:keys [id stream idx] :as entry}]
                 (let [identifier (if id
                                    {:id id}
                                    {:stream stream})]
                  (cond-> retval
                    (not (contains? retval identifier))
                    (assoc identifier idx))))
               {})
       (group-by second)
       (map (fn [[k v]]
              [k (-> (map first v)
                     set)]))
       (into {})))


(comment

      (->> access-pattern
         (reduce (fn [retval {:keys [id stream allocation-events]}]
                   (let [identifier (if id
                                      {:id id}
                                      {:stream stream})]
                    (cond-> retval
                      (contains? allocation-events :allocated)

                      (contains? allocation-events :deallocated)
                      ((fn [retval]
                         (let [[pools available] retval
                               pool-id (->> pools
                                            (filter #(contains? (second %) identifier))
                                            ffirst)]
                           [pools (conj available pool-id)]))))))
                 [{} #{}])))


(defn generate-traversal-buffer-pools
  "Generate a set of pools to use for the buffers.  The traversal system guarantees that it is safe to use the pool
with a reshape to the buffer shape in question."
  [traversal]
  (let [access-pattern
        (->> (concat (map (fn [{:keys [incoming outgoing] :as entry}]
                            {:buffers (->> (concat incoming outgoing)
                                           (map #(assoc % :usage #{:buffer})))})
                          (:forward traversal))
                     (map (fn [{:keys [incoming outgoing] :as entry}]
                            {:buffers (->> (concat incoming outgoing)
                                           (map #(assoc % :usage #{:buffer :gradient})))})
                          (:backward traversal)))
             (map-indexed (fn [idx buf-entry]
                            (assoc buf-entry :idx idx)))
             vec)
        raw-access-pattern (->> access-pattern
                                (mapcat (fn [{:keys [buffers idx]}]
                                          (map #(assoc % :idx idx) buffers))))
        first-access-map (record-first-seen-buffer-id-index raw-access-pattern)
        last-access-map (record-first-seen-buffer-id-index (reverse raw-access-pattern))]
    {:access-pattern access-pattern
     :pools
     (->> access-pattern
          (mapv (fn [{:keys [buffers idx] :as entry}]
                  (assoc entry
                         :buffers
                         (mapv (fn [{:keys [id stream] :as buf-entry}]
                                 (let [identifier (if id {:id id} {:stream stream})]
                                   (cond-> (assoc buf-entry :allocation-events #{})
                                     (get-in first-access-map [idx identifier])
                                     (update :allocation-events conj :allocated)
                                     (get-in last-access-map [idx identifier])
                                     (update :allocation-events conj :deallocated))))
                               buffers))))
          ;;Now each buffer entry can state definitively
          (reduce (fn [retval {:keys [buffers] :as entry}]
                    ;;Consider all allocation events for the function first
                    (let [retval (reduce (fn [retval {:keys [id stream allocation-events usage]}]
                                           (let [identifier (if id {:id id} {:stream stream})
                                                 [pools available] retval
                                                 [pool-id available] (if-let [pool-id (first available)]
                                                                       [pool-id (disj available pool-id)]
                                                                       [(count pools) available])]
                                             [(update pools pool-id #(assoc % identifier usage)) available]))
                                         retval
                                         (filter #(contains? (get % :allocation-events) :allocated) buffers))]
                      ;;Consider the deallocation events second
                      (reduce (fn [retval {:keys [id stream allocation-events usage]}]
                                (let [identifier (if id {:id id} {:stream stream})
                                      [pools available] retval
                                      pool-id (->> pools
                                                   (filter #(contains? (second %) identifier))
                                                   ffirst)]
                                  [(update pools pool-id
                                           #(assoc % identifier usage))
                                   (conj available pool-id)]))
                              retval
                              (filter #(contains? (get % :allocation-events) :deallocated) buffers))))
                  [{} #{}])
          first)}))
