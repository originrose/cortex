(ns cortex.compute.batching-system
  (:require [cortex.compute.driver :as drv]
            [cortex.compute.math :as math]
            [think.datatype.core :as dtype]
            [cortex.dataset :as ds]
            [cortex.loss :as loss]
            [clojure.set :as c-set]
            [clojure.core.matrix :as m]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* true)

(def batch-types [:training
                  :cross-validation
                  :holdout
                  :all])


(defprotocol PBatchingSystem
  (add-streams
    [bs batch-map]
    "Add streams to the batching system.  If the system has a matching entry,
    checks all data in the batch map, ensures the allocated size matches the incoming
    data size and that the datatypes map.  If not, a new entry is created.
    Returns a new batching system.")

  (get-batches
    [bs batch-map-sequence upload-output-buffers?]
    "Returns a map of stream-name->array where array has 2 dimensions;
    [batch-size item-total-size]"))


(defn- batch-buffers
  [backend size-entry & [batch-size]]
  (let [driver (drv/get-driver backend)
        datatype (get size-entry :datatype (dtype/get-datatype backend))
        batch-size (or batch-size 1)
        item-size (get size-entry :size)
        size item-size
        device-array (math/new-array driver
                                   (drv/get-stream backend)
                                   datatype
                                   [size]
                                   batch-size)
        host-buffer (drv/allocate-host-buffer driver (* size batch-size) datatype)]
    {:device-array device-array
     :host-buffer host-buffer}))


(defrecord DatasetBatchingSystem [backend ^long batch-size stream->batch-info-map]
  PBatchingSystem
  (add-streams [bs batch-map]
    (assoc bs :stream->batch-info-map
      (merge stream->batch-info-map
        (->> batch-map
             (map (fn [[k v]]
                    (let [v (if (map? v)
                              v
                              {:data v})
                          data-size (long (m/ecount (get v :data)))
                          item-size (quot data-size batch-size)]
                      (when-not (= 0 (rem data-size (long batch-size)))
                        (throw (ex-info "Data coming from batch is not multiple of batch-size"
                                        {:data-size data-size
                                         :batch-size batch-size
                                         :stream k})))
                      (if-let [existing (get stream->batch-info-map k)]
                        (do
                          (let [incoming-size
                                (quot (-> (get-in existing [:batch-buffers :host-buffer])
                                          (m/ecount))
                                      batch-size)]
                            (when-not (= incoming-size
                                         item-size)
                              (throw (ex-info "Incoming data size mismatch from expected size"
                                              {:incoming-size item-size
                                               :expected-size incoming-size
                                               :stream k}))))
                          [k existing])
                        [k
                         (assoc (dissoc v :data)
                                :batch-buffers (batch-buffers backend
                                                              (assoc v :size item-size)
                                                              batch-size))]))))
             (into {})))))

  (get-batches [bs batch-map-sequence required-keys]
    (when (= 0 (count required-keys))
      (throw (ex-info "Batching system did not find any keys to upload"
                      {:batch-info-map (->> (mapv (fn [[k v]]
                                                    [k
                                                     (dissoc v :batch-buffers)]))
                                            (into {}))
                       :required-keys (vec required-keys)})))
    (map (fn [batch-map]
           (when-not (every? #(contains? batch-map %) required-keys)
             (throw (ex-info "Network batching Missing streams:"
                             {:dataset-streams (keys batch-map)
                              :network-streams required-keys})))
           (->> required-keys
                (map (fn [stream]
                       (let [{:keys [batch-buffers]} (get stream->batch-info-map stream)
                             {:keys [device-array host-buffer]} batch-buffers
                             stream-entry (get batch-map stream)
                             stream-data (if (map? stream-entry)
                                           (get stream-entry :data)
                                           stream-entry)]
                         (try
                           (let [item-count (->> (dtype/copy-raw->item!  stream-data host-buffer 0)
                                                 second)
                                 host-count (m/ecount host-buffer)]
                             (when-not (= item-count (m/ecount host-buffer))
                               (throw (ex-info "Failed to copy correct number of items into buffer"
                                               {:copy-count item-count
                                                :buffer-size (m/ecount host-buffer)}))))
                           (catch Exception e
                             (throw (ex-info "Failed to load stream entry:"
                                             {:stream stream
                                              :error e
                                              :buffer-size (m/ecount host-buffer)
                                              :data-size (m/ecount stream-data)}))))
                         (drv/copy-host->device (drv/get-stream backend) host-buffer 0
                                                (math/device-buffer device-array) 0
                                                (m/ecount host-buffer))
                         [stream device-array])))
                (into {})))
         batch-map-sequence)))


(defn batching-system
  [backend stream-map batch-size]
  (->DatasetBatchingSystem backend batch-size
    (->> stream-map
         (map (fn [[k v]]
                [k (assoc v :batch-buffers
                            (batch-buffers backend v batch-size))]))
         (into {}))))

