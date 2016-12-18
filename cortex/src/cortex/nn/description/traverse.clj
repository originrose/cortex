(ns cortex.nn.description.traverse)


(defn create-forward-traversal
  "A forward traversal is a linear dfs order sequence.
There is an optional argument to remove nodes of a particular type from
the traversal.

Each item in the sequence is a map of:
{:incoming ()
 :id
 :outgoing ()
}"
  [{:keys [layer-graph] :as built-network} remove-type-set]
  (let [{:keys [nodes edges roots]} layer-graph
        remove-id-set (->> (filter #(contains? remove-type-set
                                               (get % :type)) nodes)
                           (map :id)
                           set)
        edges (if (empty? remove-id-set)
                edges
                (let [alias-map (->> (map (fn [[top bottom]]
                                            (when (contains? remove-id-set bottom)
                                              [bottom top]))
                                          edges)
                                     (into {}))]
                  (->> (map (fn [[top bottom]]
                              (when-not (contains? remove-id-set bottom)
                                [(get alias-map top top) bottom]))
                            edges)
                       (remove nil?))))
        parent->child-map (-> (->> (group-by first edges)
                                   (map (fn [[k v]] [k (distinct (map second v))]))
                                   (into {}))
                              (assoc :roots roots))
        child->parent-map (->> (group-by second edges)
                               (map (fn [[k v]] [k (distinct (map first v))]))
                               (into {}))]
    (->> (tree-seq #(contains? parent->child-map %)
                   parent->child-map
                   :roots)
         (drop 1)

         (map (fn [id]
                {:incoming (get child->parent-map id)
                 :id id
                 :outgoing (get parent->child-map id)})))))



(defn reverse-forward-traversal
  "See create-forward-traversal.  Reverse of same sequence."
  [forward-traversal]
  (->> forward-traversal
       reverse
       (map (fn [{:keys [incoming outgoing] :as traverse-item}]
              (assoc traverse-item
                     :incoming outgoing
                     :outgoing incoming)))))

;;These nodes do not alter data passing through them.
(def identity-nodes {:input :output})


(defn- create-id->node-map
  [nodes]
  (->> (group-by :id nodes)
       (map (fn [[k v]] [k (first v)]))
       (into {})))


(defn network->gradient-descent
  "Given network create master buffer list,
two traversals (forward,backward)
and input and output buffer lists.
Each traversal is sequence of maps like in create-forward-traversal
exception the incoming and outgoing ids are buffer ids.
{:buffers map id->{:buffer-size}
 :forward where incoming/outgoing maps to buffer id
 :backward where incoming/outgoing maps to buffer id}"
  [built-network & {:keys [remove-type-set]}]
  (let [forward-traversal (create-forward-traversal built-network remove-type-set)
        {:keys [nodes]} (get built-network :layer-graph)
        id->node-map (create-id->node-map nodes)
        id->outgoing #(map (fn [{:keys [id] :as item}]
                             (assoc item :outgoing [id]))
                         %)]
    {:buffers (reduce (fn [buffer-map {:keys [incoming id outgoing]}]
                        (assoc buffer-map id {:buffer-size (get-in id->node-map
                                                                   [id :output-size])}))
                      {}
                      forward-traversal)
     :forward (id->outgoing forward-traversal)
     :backward (id->outgoing (reverse-forward-traversal forward-traversal))}))


(defn- allocate-buffer
  "Assumption is that the free buffers are sorted by buffer-size"
  [id output-size free-buffers]
  (let [compare-condition #(> (get % :buffer-size) output-size)
        next-buffer (first (filter compare-condition
                                   free-buffers))]
    (if next-buffer
      [next-buffer (remove compare-condition free-buffers)]
      (let [next-buffer (or (last free-buffers)
                            {:id id})]
        [(assoc next-buffer :buffer-size output-size) (drop-last free-buffers)]))))


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
   free-buffers in-flight-buffers id->node-map]
  (when traverse-item
    (let [{:keys [incoming id outgoing]} traverse-item
          [next-buffer free-buffers] (allocate-buffer id
                                                      (get-in id->node-map
                                                              [id :output-size])
                                                      (sort-by :buffer-size
                                                               free-buffers))
          [next-free in-flight-buffers] (free-in-flight-buffers incoming in-flight-buffers)
          in-flight-buffers (assoc in-flight-buffers id (assoc next-buffer
                                                               :ref-count
                                                               (count outgoing)))
          all-buffers (assoc all-buffers
                             (get next-buffer :id)
                             next-buffer)]
      (cons
       [{:id id :outgoing (get next-buffer :id)} all-buffers]
       (lazy-seq (forward-traversal->inference (first forward-traverse-seq)
                                               (rest forward-traverse-seq)
                                               all-buffers (concat free-buffers
                                                                   next-free)
                                               in-flight-buffers id->node-map))))))


(defn network->inference
  "Similar to network->gradient-descent however in this case we have the option
  of optimising for memory which means we can aggressively reuse buffers *or*
  optimising for speed in which case the result is the forward pass of gradient descent
  and we expect implementations to have multiple batches in flight simultaneously.  We
  default to optimising for memory because this avoids OOM situations with large networks."
  [{:keys [layer-graph] :as built-network} & {:keys [optimise-type remove-type-set]
                                              :or {optimise-type :memory
                                                   remove-type-set #{:dropout}}}]

  (if (= optimise-type :memory)
    (let [basic-forward (create-forward-traversal built-network remove-type-set)
          forward-traverse-seq (forward-traversal->inference (first basic-forward)
                                                             (rest basic-forward)
                                                             {}
                                                             []
                                                             {}
                                                             (create-id->node-map
                                                              (get layer-graph :nodes)))]
      {:forward (map first forward-traverse-seq)
       :buffers (second (last forward-traverse-seq))})
    (-> (network->gradient-descent built-network :remove-type-set remove-type-set)
        (dissoc :backward))))
