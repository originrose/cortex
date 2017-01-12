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
            [cortex.loss :as cortex-loss]
            [clojure.set :as c-set]
            [clojure.core.matrix.macros :refer [c-for]]
            [think.compute.driver :as drv]
            [think.datatype.core :as dtype]
            [think.compute.math :as math]
            [think.compute.nn.cpu-backend :as cpu-backend]
            [cortex.nn.network :as network]))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  Bind/save functionality
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn- allocate-l2-temp-data
  [weights backend]
  (let [weight-shape (m/shape weights)]
   {:weight-temp (backend/new-array backend weight-shape)
    :weight-magnitude-temp (backend/new-array backend
                                              [(first weight-shape)])
    :ones-vec (backend/allocate-ones backend (second weight-shape))}))

(defn is-l2-max-constraint-valid?
  [parameter]
  (and (> (get parameter :l2-max-constraint 0.0) 0.0)
       (= :weight (get parameter :type))))


(defn- bind-node-parameter-buffers
  [compute-buffers node network backend gradients? numeric-gradients?]
  (let [driver (drv/get-driver backend)
        datatype (dtype/get-datatype backend)
        alloc-host (fn [elem-count]
                     (drv/allocate-host-buffer driver elem-count datatype))]
    (reduce (fn [compute-buffers {:keys [key non-trainable? buffer-id] :as parameter}]
              (let [gradients? (and (not non-trainable?) gradients?)
                    numeric-gradients? (and (not non-trainable?) numeric-gradients?)
                    l2-max-constraint (double (get parameter :l2-max-constraint 0.0))]
                (update compute-buffers buffer-id
                        (fn [compute-buffer]
                          (or compute-buffer
                              (let [graph-buffer (get parameter :buffer)]
                                (cond-> {:buffer (backend/array backend graph-buffer)}
                                  gradients?
                                  (assoc :gradient (backend/new-array backend
                                                                      (m/shape graph-buffer)))
                                  numeric-gradients?
                                  (assoc :numeric-gradient (alloc-host (m/ecount graph-buffer))
                                         :host-buffer (alloc-host (m/ecount graph-buffer)))
                                  (is-l2-max-constraint-valid? parameter)
                                  (merge (allocate-l2-temp-data graph-buffer backend)))))))))
            compute-buffers
            (network/get-node-parameters network (get node :id)))))


(defn- create-batching-system
  [backend built-network batch-size]
  (batching-system/create backend
                          (traverse/network->stream->size-map built-network)
                          batch-size))


(defn- get-node-parameters
  "Get a combined form of the node parameters"
  [network id]
  (->> (network/get-node-parameters network id)
       (map (fn [{:keys [buffer-id key] :as parameter}]
              [key
               (merge parameter (get-in network [:compute-binding
                                                 :parameter-buffers
                                                 buffer-id]))]))
       (into {})))


(defn- load-training-parameters
  [network]
  (->> (get-in network [:compute-binding :id->node-map])
       keys
       (mapcat #(map second
                 (get-node-parameters network %)))
       (remove #(get % :non-trainable?))
       vec))


(defn- bind
  [backend-fn {:keys [batch-size layer-graph traversal] :as built-network}
   {:keys [gradients? numeric-gradients?] :as options}]
  (let [backend (backend-fn)
        id->node-map (get layer-graph :id->node-map)
        traverse-type (get traversal :type)
        gradients? (or gradients? (= traverse-type :training))
        driver (drv/get-driver backend)
        datatype (dtype/get-datatype backend)
        alloc-host (fn [elem-count]
                     (drv/allocate-host-buffer driver elem-count datatype))
        backward-buffers (if gradients?
                           (traverse/network->backward-buffer-set built-network)
                           #{})
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
                                                           backend gradients?
                                                           numeric-gradients?))))))
         (get-in built-network [:compute-binding])
         (get traversal :forward))
        compute-binding
        (reduce
         (fn [compute-binding buffer-key]
           (update-in compute-binding [:traversal-buffers buffer-key]
                      (fn [buffer]
                        (or buffer
                            (let [buffer-size (get-in traversal [:buffers buffer-key :size])
                                  gradients? (and gradients?
                                                  (contains? backward-buffers buffer-key))
                                  numeric-gradients? (and numeric-gradients?
                                                          (contains? backward-buffers
                                                                     buffer-key))]
                              (cond-> {:buffer (backend/new-array backend [buffer-size]
                                                                  batch-size)}
                                gradients?
                                (assoc :gradient (backend/new-array backend [buffer-size]
                                                                    batch-size))
                                numeric-gradients?
                                (assoc :numeric-gradient (alloc-host (* buffer-size batch-size))
                                       :host-buffer (alloc-host (* buffer-size
                                                                   batch-size)))))))))
         compute-binding
         (keys (get traversal :buffers)))
        network (assoc built-network
                       :compute-binding
                       (assoc compute-binding
                              :backend backend
                              :batching-system (create-batching-system backend built-network
                                                                       batch-size)))
        trainable-parameters (load-training-parameters network)
        trainable-param-count (->> trainable-parameters
                                   (map (comp m/ecount :buffer))
                                   (apply +))
        l1-buffer-size (long (->> trainable-parameters
                                  (filter #(> (get % :l1-regularization 0) 0))
                                  (map #(m/ecount (get % :buffer)))
                                  (apply max 0)))
        network (if (> l1-buffer-size 0)
                  (assoc-in network [:compute-binding :l1-buffer]
                            (drv/allocate-device-buffer driver l1-buffer-size datatype))
                  network)]
    (-> network
        (assoc-in [:compute-binding :optimiser]
                  (when-let [optimiser (get traversal :optimiser)]
                    (compute-optimise/create-compute-optimiser backend
                                                               optimiser
                                                               trainable-param-count)))
     (assoc-in [:compute-binding :trainable-parameters] trainable-parameters))))


(defn- save
  [network {:keys [save-gradients?]}]
  (let [backend (get-in network [:compute-binding :backend])
        core-m (fn [data]
                 (when data
                   (backend/to-core-matrix backend data)))
        ->doubles (fn [host-buffer]
                    (when host-buffer
                     (let [retval (double-array (m/ecount host-buffer))]
                       (dtype/copy! host-buffer 0 retval 0 (m/ecount host-buffer))
                       retval)))]
    (-> network
        (update-in [:layer-graph :buffers]
                   (fn [buffers]
                     (reduce
                      (fn [buffers [buf-id {:keys [buffer gradient numeric-gradient]}]]
                        (update buffers buf-id
                                (fn [result-buffer]
                                  (cond-> (assoc result-buffer :buffer (core-m buffer))
                                    (and save-gradients? gradient)
                                    (assoc :gradient (core-m gradient)
                                           :numeric-gradient (->doubles numeric-gradient))))))
                      buffers
                      (get-in network [:compute-binding :parameter-buffers]))))
        (assoc-in [:traversal :buffers]
                  (if save-gradients?
                    (reduce (fn [buffers [buf-id {:keys [buffer gradient numeric-gradient]}]]
                              (update buffers buf-id
                                      #(assoc
                                        %
                                        :buffer (core-m buffer)
                                        :gradient (core-m gradient)
                                        :numeric-gradient (->doubles numeric-gradient))))
                            {}
                            (get-in network [:compute-binding :traversal-buffers]))
                    (get-in network [:traversal :buffers])))
        (dissoc :compute-binding))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Specific traversal implementation
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
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
   ;;Raw forward is used for gradient checking and thus does not use the prepare step.
   :raw-forward {:pass-functions [compute-protocols/forward]
                 :buffer-type :buffer
                 :input-key :input-stream
                 :traversal-key :forward}
   :backward {:pass-functions [compute-protocols/backward]
              :buffer-type :gradient
              :input-key :output-id
              :traversal-key :backward}})


(defn- find-buffers
  [traversal-buffers buffer-ids]
  (mapv traversal-buffers buffer-ids))


(defn- map-pass-to-buffers
  "Create a new pass with items mapped to buffers."
  [network id->input-buffer-map pass-direction]
  (let [{:keys [traversal-key buffer-type input-key]} (get pass-metadata pass-direction)
        traversal-pass (get-in network [:traversal traversal-key])
        backend (get-in network [:compute-binding :backend])
        traversal-buffers (->> (get-in network [:compute-binding :traversal-buffers])
                               (map (fn [[map-key buffer-entry]]
                                      ;;Assoc the input buffers into
                                      ;;the appropriate spots if they
                                      ;;are passed in.
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


(defn- print-traversal-buffers
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
  [network id->buffer-map pass-direction]
  (let [[network mapped-pass] (map-pass-to-buffers network
                                                   id->buffer-map
                                                   pass-direction)]
    (reduce (fn [network pass-function]
              (->> mapped-pass
                   (map (fn [{:keys [incoming id outgoing]}]
                          (pass-function
                           (get-in network [:compute-binding :id->node-map id])
                           (get-node-parameters network id)
                           incoming outgoing)))
                   dorun)
              network)
            network
            (get-in pass-metadata [pass-direction :pass-functions]))))


(defn- load-id->input-map
  [network id->input-map]
  (let [batch-size (get network :batch-size)
        backend (get-in network [:compute-binding :backend])]
    (->> id->input-map
         (map (fn [[k v]]
                [k (backend/array backend v batch-size)]))
         (into {}))))


(defn- traverse
  "Expectiation is that the id->input-map has buffers that aren't already uploaded
to the device."
  [network id->input-map pass-direction]
  (do-traverse network
               (load-id->input-map network id->input-map)
               pass-direction))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;Training
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn- get-output-bindings
  [network]
  (->> (traverse/get-output-bindings network)
       (filter #(get % :stream))
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


(defn- get-input-bindings
  [network]
  (->> (traverse/get-input-bindings network)
       (filter #(get % :stream))
       (map (fn [{:keys [stream node-id] :as entry}]
              (assoc entry
                     :buffers
                     (get-in network [:compute-binding
                                      :traversal-buffers
                                      {:input-stream stream}])
                     :size (get-in network [:layer-graph
                                            :id->node-map
                                            node-id
                                            :input-size]))))))


(defn- apply-l2-max-constraint
  [backend {:keys [weight-temp weight-magnitude-temp ones-vec buffer l2-max-constraint]}]
  (when l2-max-constraint
    (let [weight-ecount (long (math/ecount buffer))
          [num-w-rows num-w-cols] (math/shape-2d buffer)]
      (backend/assign! backend weight-temp buffer)
      (math/elem-mul (drv/get-stream backend)
                     1.0 (math/device-buffer buffer) 1
                     (math/device-buffer weight-temp) 1
                     (math/device-buffer weight-temp) 1)
      (math/gemv (drv/get-stream backend) false num-w-rows num-w-cols
                 1.0 (math/device-buffer weight-temp) num-w-cols
                 (math/device-buffer ones-vec) 1
                 0.0 (math/device-buffer weight-magnitude-temp) 1)
      (math/l2-constraint-scale (drv/get-stream backend)
                                (math/device-buffer weight-magnitude-temp) 1
                                l2-max-constraint)
      (math/mul-rows (drv/get-stream backend) num-w-rows num-w-cols
                     (math/device-buffer buffer) num-w-cols
                     (math/device-buffer weight-magnitude-temp) 1
                     (math/device-buffer buffer) num-w-cols))))

(defn- optimise-network
  [network parameters]
  (let [backend (get-in network [:compute-binding :backend])
        stream (drv/get-stream backend)
        driver (drv/get-driver backend)
        optimiser (compute-optimise/batch-update (get-in network [:compute-binding
                                                                  :optimiser]))
        buffer-alpha (/ 1.0 (double (get network :batch-size)))]
    (reduce (fn [offset {:keys [buffer gradient
                                learning-attenuation non-trainable?] :as parameter}]
              (let [elem-count (long (m/ecount buffer))
                    l2-reg (double (get parameter :l2-regularization 0))
                    l1-reg (double (get parameter :l1-regularization 0))
                    l2-max-constraint (double (get parameter :l2-max-constraint 0))
                    ;;For some things it is easier to just
                    ;;work at the flat buffer level and
                    ;;not at the device array level.
                    gradient-buf (math/device-buffer gradient)
                    param-buf (math/device-buffer buffer)]
                (when-not non-trainable?
                  (when (> l2-reg 0.0)
                    (math/sum stream l2-reg param-buf 1.0 gradient-buf gradient-buf))
                  (when (> l1-reg 0.0)
                    (let [l1-buffer (get-in network [:compute-binding :l1-buffer])]
                      (when-not (and l1-buffer
                                     (>= (long (m/ecount l1-buffer))
                                         elem-count))
                        (throw (ex-info "l1 reg buffer missing or too small"
                                        {:l1-buffer-size (m/ecount l1-buffer)})))
                      (let [l1-buffer (drv/sub-buffer driver l1-buffer 0 elem-count)]
                        ;;The derivative of the absolute value function is either -1 or 1
                        ;;depending on the value of the parameter.
                        (math/select stream param-buf l1-buffer -1 1)
                        ;;Add derivatives to gradient
                        (math/sum stream l1-reg l1-buffer 1.0 gradient-buf gradient-buf))))
                  (compute-optimise/compute-parameters! optimiser
                                                        (* buffer-alpha learning-attenuation)
                                                        offset gradient buffer)
                  (when (is-l2-max-constraint-valid? parameter)
                    (apply-l2-max-constraint backend parameter)))
                (when gradient
                  (drv/memset stream gradient-buf 0 0 elem-count))
                (+ offset elem-count)))
            0
            parameters)
    (assoc-in network
              [:compute-binding :optimiser]
              optimiser)))


(defn- recur-train-sequence
  "Training is a lazy sequence of these operations."
  [network parameters output-bindings optimise? batch-seq]
  (when-let [stream->buffer-map (first batch-seq)]
    ;;Sometimes you have to print the entire batch out to see what is going on.
    (let [backend (get-in network [:compute-binding :backend])
          stream (drv/get-stream backend)
          driver (drv/get-driver backend)
          ten-doubles #(vec (take 10 (backend/to-double-array backend %)))]
      (comment
        (clojure.pprint/pprint (mapv (fn [[k v]]
                                       [k (ten-doubles v)])
                                     stream->buffer-map)))
      (let [network (do-traverse network stream->buffer-map :forward)
            backend (get-in network [:compute-binding :backend])
            stream (drv/get-stream backend)
            driver (drv/get-driver backend)]
        (doseq [{:keys [buffers loss stream] :as entry} output-bindings]
          (let [{:keys [buffer gradient]} buffers
                answer (get stream->buffer-map stream)]
            ;;If the entire backward pass was nixed then there was no
            (when gradient
              (compute-loss/compute-loss-gradient loss backend buffer answer gradient))
            (comment
              (clojure.pprint/pprint {:loss loss
                                      :buffer (ten-doubles buffer)
                                      :answer (ten-doubles answer)
                                      :gradient (ten-doubles gradient)}))))
        ;;Backward traverse uses existing buffers and doesn't need any id->buffer remapping
        (do-traverse network {} :backward)
        (let [network
              (if optimise?
                (optimise-network network parameters)
                network)]
          (cons network
                (lazy-seq (recur-train-sequence network parameters
                                                output-bindings optimise?
                                                (rest batch-seq)))))))))

(defn- train-sequence
  [network batch-map-sequence options]
  (let [bs (get-in network [:compute-binding :batching-system])
        backend (get-in network [:compute-binding :backend])
        output-bindings (get-output-bindings network)
        ;;These are the things we are ultimately optimizing
        parameters (get-in network [:compute-binding :trainable-parameters])
        ;;The buffers do not change going backward so we can pre-map this pass.
        [network backward-mapped-pass] (map-pass-to-buffers network
                                                            {}
                                                            :backward)]
    (->> (batching-system/get-batches bs batch-map-sequence true)
         (recur-train-sequence network parameters output-bindings true))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;  Inference
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn- create-infer-seq-support-data
  [network]
  (let [batch-size (long (get network :batch-size))
        backend (get-in network [:compute-binding :backend])
        driver (drv/get-driver backend)
        stream (drv/get-stream backend)
        datatype (dtype/get-datatype backend)
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
    {:output-bindings output-bindings
     :copy-fn copy-fn}))


(defn- do-infer-seq
  [network {:keys [copy-fn output-bindings] :as support-data} pass-direction batches]
  (let [batch-size (long (get network :batch-size))
        backend (get-in network [:compute-binding :backend])
        driver (drv/get-driver backend)
        stream (drv/get-stream backend)
        datatype (dtype/get-datatype backend)]
   (map (fn [stream->buffer-map]
          (let [network (do-traverse network stream->buffer-map pass-direction)]
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
                          ;;Wait for the network to finish the inference traversal
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
                 (into {}))))
        batches)))


(defn- batch-infer
  [network batch-map-sequence options]
  (let [bs (get-in network [:compute-binding :batching-system])
        support-data (create-infer-seq-support-data network)]
    (->> (batching-system/get-batches bs batch-map-sequence false)
         (do-infer-seq network support-data :inference))))


(defn- generate-numeric-gradients
  [context network stream->input-map epsilon]
  (let [output-bindings (get-output-bindings network)
        stream->data-map (load-id->input-map network stream->input-map)
        ;;Generate all of the calculated gradients.
        parameters (get-in network [:compute-binding :trainable-parameters])
        ;;This calls prepare-forward exactly once and does one forward
        ;;plus backward and loss gradient to generate calculated gradients
        network (first (recur-train-sequence network
                                             parameters
                                             output-bindings
                                             false
                                             [stream->data-map]))
        ;;generate a sequence of buffers in order to generate the numeric gradients.
        numeric-buffers (concat (->> (get-input-bindings network)
                                     (map (fn [{:keys [stream] :as entry}]
                                            (merge (dissoc entry :buffers)
                                                   (get entry :buffers)))))
                                (remove #(get % :non-trainable?) parameters))
        epsilon (double epsilon)
        support-data (create-infer-seq-support-data network)
        node-id->output-binding (->> output-bindings
                                     (map (fn [{:keys [node-id] :as entry}]
                                            [node-id entry]))
                                     (into {}))
        stream (drv/get-stream (get-in network [:compute-binding :backend]))
        forward-fn (fn [param-value host-buffer device-buffer elem-count idx]
                     (dtype/set-value! host-buffer idx param-value)
                     (drv/copy-host->device stream host-buffer 0 device-buffer 0 elem-count)
                     ;;Raw-forward is used here to avoid calling prepare-forward again.  But this
                     ;;is not an inference pass; it is an actual forward pass.
                     (first (do-infer-seq network support-data :raw-forward [{}])))]
    (doseq [{:keys [buffer numeric-gradient host-buffer] :as entry} numeric-buffers]
      (let [device-buffer (math/device-buffer buffer)]
       (when-not (and numeric-gradient host-buffer)
         (throw (ex-info "failed to allocate appropriate buffers for numeric gradients."
                         {:buffer-keys (keys entry)})))
       (let [elem-count (m/ecount buffer)]
         (drv/copy-device->host stream device-buffer 0 host-buffer 0 elem-count)
         (drv/wait-for-event (drv/create-event stream))
         (doseq [idx (range elem-count)]
           (let [param-value (double (dtype/get-value host-buffer idx))
                 positive (forward-fn (+ param-value epsilon) host-buffer device-buffer elem-count idx)
                 negative (forward-fn (- param-value epsilon) host-buffer device-buffer elem-count idx)
                 gradient (->> (keys positive)
                               (map (fn [node-id]
                                      (let [pos-data (get positive node-id)
                                            neg-data (get negative node-id)
                                            {:keys [loss stream output-size]}
                                            (get node-id->output-binding node-id)
                                            ;;Partition stream into batches
                                            stream-data (->> (get stream->input-map stream)
                                                             m/eseq
                                                             (partition output-size))]
                                        (->>
                                         (map (fn [pos neg answer]
                                                (/
                                                 (- (double (cortex-loss/loss loss pos answer))
                                                    (double (cortex-loss/loss loss neg answer)))
                                                 (* 2 epsilon)))
                                              pos-data neg-data stream-data)
                                         (apply +)))))
                               (apply +))]
             (dtype/set-value! host-buffer idx param-value)
             ;;Reset device buffer to original value.
             (drv/copy-host->device stream host-buffer 0 device-buffer 0 elem-count)
             (dtype/set-value! numeric-gradient idx gradient))))))
    network))


(defrecord ComputeExecutionContext [backend-fn]
  cp/PExecutionContext
  (bind-to-network [context built-network options]
    (bind backend-fn built-network options))
  (train-batch-sequence [context built-network batch-map-sequence options]
    (train-sequence built-network batch-map-sequence options))
  (infer-batch-sequence [context built-network batch-map-sequence options]
    (batch-infer built-network batch-map-sequence options))
  (save-to-network [context built-network options]
    (save built-network options))
  (traverse [context bound-network id->input-map direction]
    (traverse bound-network id->input-map direction))
  (generate-numeric-gradients [context built-network stream->data-map epsilon]
    (generate-numeric-gradients context built-network stream->data-map epsilon)))


(defn create-context
  ([backend-fn]
   (->ComputeExecutionContext backend-fn))
  ([]
   (create-context #(cpu-backend/create-cpu-backend))))
