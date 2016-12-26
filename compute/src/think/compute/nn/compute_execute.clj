(ns think.compute.nn.compute-execute
  (:require [cortex.nn.execute :as execute]
            [cortex.nn.layers :as layers]
            [cortex.nn.traverse :as traverse]
            [think.compute.nn.layers :as compute-layers]
            [think.compute.optimise :as compute-optimise]
            [think.compute.nn.backend :as backend]
            [cortex.nn.protocols :as cp]
            [think.compute.nn.protocols :as compute-protocols]
            [clojure.core.matrix :as m]
            [think.resource.core :as resource]
            [think.compute.batching-system :as batching-system]
            [think.compute.loss :as compute-loss]
            [clojure.set :as c-set]
            [clojure.core.matrix.macros :refer [c-for]]
            [think.compute.driver :as drv]
            [think.datatype.core :as dtype]
            [think.compute.math :as math]))


(defn- bind-node-parameter-buffers
  [compute-buffers node network backend traverse-type]
  (reduce (fn [compute-buffers {:keys [key]}]
            (let [{:keys [buffer-id] :as param-entry} (get node key)]
              (update compute-buffers buffer-id
                      (fn [compute-buffer]
                        (or compute-buffer
                            (let [graph-buffer (get-in network
                                                       [:layer-graph
                                                        :buffers
                                                        buffer-id
                                                        :buffer])]
                              (cond-> {:buffer (backend/array backend graph-buffer)}
                                (= traverse-type :training)
                                (assoc :gradient (backend/new-array backend
                                                                    (m/shape graph-buffer))))))))))
          compute-buffers
          (layers/get-parameter-descriptions node)))


(defn- create-batching-system
  [backend built-network batch-size]
  (let [bindings (traverse/get-dataset-bindings built-network)
        stream->size-map (->> bindings
                              (map (fn [{:keys [node-id dataset-stream direction]}]
                                     (when dataset-stream
                                       [dataset-stream
                                        ;;Using a set here to detect size mismatches
                                        {:size #{(get-in built-network
                                                          [:layer-graph :id->node-map node-id
                                                           (if (= direction :input)
                                                             :input-size
                                                             :output-size)])}
                                         :direction #{direction}}])))
                              (remove nil?)
                              (reduce (fn [eax [stream {:keys [size direction] :as entry}]]
                                        (-> eax
                                            (update-in [stream :size]
                                                       #(c-set/union %  size))
                                            (update-in [stream :direction]
                                                       #(c-set/union % direction))))
                                      {})
                              (map (fn [[k {:keys [size] :as entry}]]
                                     (when (> (count size) 1)
                                       (throw (ex-info "Stream is mapped to different sized nodes:"
                                                       {:stream k
                                                        :entry entry})))
                                     [k (assoc entry :size (first size))]))
                              (into {}))]
    (batching-system/create backend stream->size-map batch-size)))


(defn- bind
  [backend-fn {:keys [batch-size layer-graph traversal] :as built-network}]
  (let [backend (backend-fn)
        id->node-map (get layer-graph :id->node-map)
        traverse-type (get traversal :type)
        compute-binding
        (reduce
         (fn [compute-binding {:keys [incoming id outgoing]}]
           (let [node (get-in layer-graph [:id->node-map id])
                 node-params (layers/get-parameter-descriptions node)]
             (-> (update-in compute-binding [:id->node-map id]
                            (fn [compute-node]
                              (or compute-node
                                  (compute-layers/create backend node batch-size))))
                 (update-in [:parameter-buffers]
                            (fn [param-buffers]
                              (bind-node-parameter-buffers param-buffers node built-network
                                                           backend traverse-type))))))
         (get-in built-network [:compute-binding])
         (get traversal :forward))
        compute-binding
        (reduce
         (fn [compute-binding buffer-key]
           (update-in compute-binding [:traversal-buffers buffer-key]
                      (fn [buffer]
                        (or buffer
                            (let [buffer-size (get-in traversal [:buffers buffer-key :size])]
                              (cond-> {:buffer (backend/new-array backend [buffer-size] batch-size)}
                                (= traverse-type :training)
                                (assoc :gradient (backend/new-array backend [buffer-size] batch-size))))))))
         compute-binding
         (keys (get traversal :buffers)))]
    (assoc built-network
           :compute-binding
           (assoc compute-binding
                  :backend backend
                  :batching-system (create-batching-system backend built-network batch-size)
                  :optimiser (compute-optimise/create-compute-optimiser backend
                                                                        (get traversal :optimiser)
                                                                        (get built-network :parameter-count))))))


(defn- save
  [network {:keys [save-gradients?]}]
  (let [backend (get-in network [:compute-binding :backend])
        core-m (fn [data] (backend/to-core-matrix backend data))]
    (-> network
        (update-in [:layer-graph :buffers]
                   (fn [buffers]
                     (reduce (fn [buffers [buf-id {:keys [buffer gradient]}]]
                               (update buffers buf-id
                                       (fn [result-buffer]
                                         (cond-> (assoc result-buffer :buffer (core-m buffer))
                                           save-gradients?
                                           (assoc :gradient (core-m gradient))))))
                             buffers
                             (get-in network [:compute-binding :parameter-buffers]))))
        (assoc-in [:traversal :buffers]
                  (if save-gradients?
                    (reduce (fn [buffers [buf-id {:keys [buffer gradient]}]]
                              (update buffers buf-id
                                      #(assoc
                                        %
                                        :buffer (core-m buffer)
                                        :gradient (core-m gradient))))
                            {}
                            (get-in network [:compute-binding :traversal-buffers]))
                    (get-in network [:traversal :buffers])))
        (dissoc :compute-binding))))


(defn- find-buffers
  [traversal-buffers buffer-ids]
  (mapv traversal-buffers buffer-ids))


(def pass-metadata
  {:inference {:pass-functions [compute-protocols/infer]
               :buffer-type :buffer
               :input-key :input-stream
               :traversal-key :forward}
   :forward {:pass-functions [compute-protocols/prepare-forward!
                             compute-protocols/forward]
             :buffer-type :buffer
             :input-key :input-stream
             :traversal-key :forward}
   :backward {:pass-functions [compute-protocols/backward]
             :buffer-type :gradient
             :input-key :output-id
             :traversal-key :backward}})


(defn- map-pass-to-buffers
  "Create a new pass with items mapped to buffers."
  [network id->input-buffer-map pass-direction]
  (let [{:keys [traversal-key buffer-type input-key]} (get pass-metadata pass-direction)
        traversal-pass (get-in network [:traversal traversal-key])
        backend (get-in network [:compute-binding :backend])
        traversal-buffers (->> (get-in network [:compute-binding :traversal-buffers])
                               (map (fn [[map-key buffer-entry]]
                                      ;;Assoc the input buffers into the appropriate spots if they are
                                      ;;passed in.
                                      (let [input-buffer (get id->input-buffer-map
                                                              (get map-key input-key))]
                                        (if input-buffer
                                          [map-key (assoc buffer-entry buffer-type input-buffer)]
                                          [map-key buffer-entry]))))
                               (into {}))
        buffer-resolve (partial find-buffers traversal-buffers)]
    [(assoc-in network [:compute-binding :traversal-buffers] traversal-buffers)
     (->> traversal-pass
          (mapv (fn [{:keys [incoming outgoing] :as item}]
                  (assoc item
                         :incoming (buffer-resolve incoming)
                         :outgoing (buffer-resolve outgoing)))))]))


(defn- get-node-parameters
  "Get a combined form of the node parameters"
  [network id]
  (->> (layers/get-parameter-descriptions
        (get-in network [:layer-graph :id->node-map id]))
       (map (fn [{:keys [key]}]
              (let [node-parameter (get-in network [:layer-graph :id->node-map id key])
                    parameter-buffer (get-in network [:compute-binding
                                                      :parameter-buffers
                                                      (get node-parameter :buffer-id)])]
                [key
                 (merge node-parameter parameter-buffer)])))
       (into {})))


(defn print-traversal-buffers
  [network]
  (println "!!Traversal buffers!!")
  (let [backend (get-in network [:compute-binding :backend])
        to-double #(vec (backend/to-double-array backend %))]
   (clojure.pprint/pprint (mapv (fn [[k v]]
                                  [k {:buffer (to-double (get v :buffer))
                                      :gradient (to-double (get v :gradient))}])
                                (get-in network [:compute-binding :traversal-buffers])))
   network))


(defn- do-traverse
  [network mapped-pass pass-function]
  (->> mapped-pass
       (map (fn [{:keys [incoming id outgoing]}]
              (pass-function
               (get-in network [:compute-binding :id->node-map id])
               (get-node-parameters network id)
               incoming outgoing)))
       dorun)
  network)


(defn- traverse
  [network id->input-map pass-direction]
  (let [batch-size (get network :batch-size)
        backend (get-in network [:compute-binding :backend])
        [network mapped-pass]
        (map-pass-to-buffers network
                             (->> id->input-map
                                  (map (fn [[k v]]
                                         [k (backend/array backend v batch-size)]))
                                  (into {}))
                             pass-direction)]
    (reduce (fn [network pass-fn]
              (do-traverse network mapped-pass pass-fn))
            network
            (get-in pass-metadata [pass-direction :pass-functions]))))


(defn- get-output-bindings
  [network]
  (->> (traverse/get-dataset-bindings network)
       (filter #(and (= :output (get % :direction))
                     (get % :dataset-stream)))
       (map (fn [{:keys [node-id] :as entry}]
              (assoc entry
                     :buffers
                     (get-in network [:compute-binding
                                      :traversal-buffers
                                      {:output-id node-id}])
                     :output-size (get-in network [:layer-graph
                                                   :id->node-map
                                                   node-id
                                                   :output-size]))))))


(defn- recur-train-sequence
  [network parameters backward-mapped-pass output-bindings batch-seq]
  (when-let [stream->buffer-map (first batch-seq)]
    ;;Sometimes you have to print the entire batch out to see what is going on.
    (comment
      (clojure.pprint/pprint (mapv (fn [[k v]]
                                    [k (vec (backend/to-double-array
                                             (get-in network [:compute-binding :backend])
                                             v))])
                                  stream->buffer-map)))
    (let [[network forward-mapped-pass] (map-pass-to-buffers network
                                                             stream->buffer-map
                                                             :forward)
          optimiser (compute-optimise/batch-update (get-in network [:compute-binding :optimiser]))
          buffer-alpha (/ 1.0 (double (get network :batch-size)))
          backend (get-in network [:compute-binding :backend])]
      (forward-traverse network forward-mapped-pass)
      (doseq [{:keys [buffers loss-function dataset-stream] :as entry} output-bindings]
        (let [{:keys [buffer gradient]} buffers
              answer (get stream->buffer-map dataset-stream)]
          (compute-loss/compute-loss-gradient loss-function backend buffer answer gradient)
          (comment
           (clojure.pprint/pprint {:loss-function loss-function
                                   :buffer (vec (take 10 (backend/to-double-array backend buffer)))
                                   :answer (vec (take 10 (backend/to-double-array backend answer)))
                                   :gradient (vec (take 10 (backend/to-double-array backend gradient)))}))))
      (backward-traverse network backward-mapped-pass)
      (reduce (fn [offset {:keys [buffers learning-attenuation]}]
                (let [{:keys [buffer-id buffer gradient]} buffers
                      elem-count (long (m/ecount buffer))]
                  (compute-optimise/compute-parameters! optimiser
                                                        (* buffer-alpha learning-attenuation)
                                                        offset gradient buffer)
                  (drv/memset (drv/get-stream backend) (math/device-buffer gradient)
                              0 0 elem-count)

                  (+ offset elem-count)))
              0
              parameters)
      (let [network (assoc-in network
                               [:compute-binding :optimiser]
                               optimiser)]
        (comment
         (clojure.pprint/pprint (mapv (fn [[k v]]
                                        [k
                                         (vec (take 10
                                                    (backend/to-double-array (get-in network
                                                                                     [:compute-binding :backend])
                                                                             (:buffer v))))])
                                      (get-in network [:compute-binding :traversal-buffers]))))
        (cons network
              (lazy-seq (recur-train-sequence network parameters backward-mapped-pass
                                              output-bindings (rest batch-seq))))))))


(defn- train-sequence
  [network batch-map-sequence options]
  (let [bs (get-in network [:compute-binding :batching-system])
        backend (get-in network [:compute-binding :backend])
        output-bindings (get-output-bindings network)
        ;;These are the things we are ultimately optimizing
        parameters (->> (get-in network [:layer-graph :id->node-map])
                        (mapcat (fn [[id node]]
                                  (map (fn [{:keys [key]}]
                                         (let [param-entry (get node key)
                                               param-buffer (get-in network [:compute-binding
                                                                             :parameter-buffers
                                                                             (get param-entry
                                                                                  :buffer-id)])]
                                           (assoc param-entry
                                                  :buffers param-buffer
                                                  :learning-attenuation (get node
                                                                             :learning-attenuation
                                                                             1.0))))
                                       (layers/get-parameter-descriptions node))))
                        vec)
        ;;The buffers do not change going backward so we can pre-map this pass.
        [network backward-mapped-pass] (map-pass-to-buffers network
                                                            {}
                                                            :backward)]
    (->> (batching-system/get-batches bs batch-map-sequence true)
         (recur-train-sequence network parameters backward-mapped-pass output-bindings))))


(defn- infer-sequence
  [network batch-map-sequence options]
  (let [bs (get-in network [:compute-binding :batching-system])
        backend (get-in network [:compute-binding :backend])
        driver (drv/get-driver backend)
        stream (drv/get-stream backend)
        datatype (dtype/get-datatype backend)
        batch-size (long (get-in network [:batch-size]))
        output-bindings (->> (get-output-bindings network)
                             (mapv (fn [{:keys [output-size] :as entry}]
                                     (assoc entry
                                            :elem-count (* batch-size (long output-size))
                                            :host-buffer
                                            (drv/allocate-host-buffer driver
                                                                      (* batch-size
                                                                         (long output-size))
                                                                      datatype)))))
        copy-fn (fn [^long idx ^long output-size host-buffer double-buffers]
                  (dtype/copy! host-buffer (* idx output-size)
                               (get double-buffers idx) 0
                               output-size))]
    (->> (batching-system/get-batches bs batch-map-sequence false)
         (map (fn [stream->buffer-map]
                (let [[network forward-mapped-pass] (map-pass-to-buffers network
                                                                          stream->buffer-map
                                                                         :forward)]
                  (forward-traverse network forward-mapped-pass)
                  (->> output-bindings
                       (map (fn [{:keys [buffers node-id output-size host-buffer elem-count]}]
                              (let [buffer (get buffers :buffer)
                                    double-buffers (->> (repeatedly batch-size
                                                                    #(double-array output-size))
                                                        vec)]
                                (drv/copy-device->host stream
                                                       (math/device-buffer buffer) 0
                                                       host-buffer 0
                                                       elem-count)
                                ;;with async copies this event is necessary.
                                (drv/wait-for-event (drv/create-event stream))
                                ;;This step is surprisingly slow if you are mixing datatypes - meaning
                                ;;if your network is running anything other than double arithmetic -
                                ;;fundamentally mixing datatypes when copying to/from nio buffers is
                                ;;very slow at the moment.
                                (if (< batch-size (.availableProcessors (Runtime/getRuntime)))
                                  (c-for [idx 0 (< idx batch-size) (inc idx)]
                                         (copy-fn idx output-size host-buffer double-buffers))
                                  (->> (pmap #(copy-fn % output-size host-buffer double-buffers)
                                             (range batch-size))
                                       dorun))
                                [node-id double-buffers])))
                       (into {}))))))))


(defrecord ComputeExecutionContext [backend-fn]
  cp/PExecutionContext
  (bind-to-network [context built-network options]
    (bind backend-fn built-network))
  (train-batch-sequence [context built-network batch-map-sequence options]
    (train-sequence built-network batch-map-sequence options))
  (infer-batch-sequence [context built-network batch-map-sequence options]
    (infer-sequence built-network batch-map-sequence options))
  (save-to-network [context built-network options]
    (save built-network options))
  (traverse [context bound-network id->input-map direction]
    (traverse bound-network id->input-map direction))
  (forward-backward-loss [context built-network
                          stream->input-map
                          node-id->loss-function-answer-map
                          epsilon]
    "Run network forward and backward like 'forward-backward' but also calculate numeric
gradients w/r/t the loss function and the provided answer.  This allows for gradient
checking."))

(defn create-context
  [backend-fn]
  (->ComputeExecutionContext backend-fn))
