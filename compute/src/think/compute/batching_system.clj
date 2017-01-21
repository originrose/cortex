(ns think.compute.batching-system
  (:require [think.compute.driver :as drv]
            [think.compute.math :as math]
            [think.datatype.core :as dtype]
            [cortex.dataset :as ds]
            [cortex.loss :as loss]
            [clojure.set :as c-set]
            [clojure.core.matrix :as m]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* true)


(def batch-types
  [:training :cross-validation :holdout :all])


(defprotocol PBatchingSystem
  (add-streams [bs stream->size-map]
    "Add streams to the batching system.  This is for late binding augmented data streams
and the size in this case *includes* the batch size if necessary.
Returns a new batching system.")

  (get-batches [bs batch-map-sequence required-keys]
    "Returns a sequence where each item of the sequence contains:
{:input-buffers - vector of buffers used for input
 :output-buffers - vector of buffers used for output
}"))

(defn- create-batch-buffers
  [backend size-entry & [batch-size]]
  (let [driver (drv/get-driver backend)
        size-entry (if-not (map? size-entry)
                     {:size size-entry}
                     size-entry)
        datatype (get size-entry :datatype (dtype/get-datatype backend))
        batch-size (or batch-size 1)
        item-size (get size-entry :size)
        size item-size]
    {:device-array (math/new-array driver
                                   (drv/get-stream backend)
                                   datatype
                                   [size]
                                   batch-size)
     :host-buffer (drv/allocate-host-buffer driver (* size batch-size) datatype)}))


(defrecord DatasetBatchingSystem [backend stream->batch-info-map]
  PBatchingSystem
  (add-streams [bs stream->size-map]
    ;;Note that for this case we assume the batch size is included in the size
    ;;if necessary.  There are some streams that are of fixed size requiredless of batch
    ;;size such as a stream aggregating label counts.
    (assoc bs :stream->batch-info-map
           (->> stream->size-map
                (reduce (fn [stream->batch-info-map [stream size]]
                          (update stream->batch-info-map stream
                                  (fn [entry]
                                    (if entry
                                      (let [declared-size (m/ecount (get-in entry [:batch-buffers :host-buffer]))
                                            size (if (map? size)
                                                   (get size :size)
                                                   size)]
                                        (when-not (= size declared-size)
                                          (throw (ex-info "Stream size does not match batching system declared size"
                                                          {:stream stream
                                                           :buffer-size declared-size
                                                           :data-size size})))
                                        entry)
                                      {:batch-buffers (create-batch-buffers backend size 1)}))))
                        stream->batch-info-map))))
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
                           (let [item-count (->> (dtype/copy-raw->item! stream-data host-buffer 0)
                                                 second)]
                             (when-not (= item-count (m/ecount host-buffer))
                               (throw (ex-info "Failed to copy correct number of items into buffer"
                                               {:copy-count item-count
                                                :buffer-size (m/ecount host-buffer)}))))
                           (catch Exception e
                             (throw (ex-info "Failed to load stream entry:"
                                             {:stream stream
                                              :buffer-size (m/ecount host-buffer)
                                              :data-size (m/ecount stream-data)}))))
                         (drv/copy-host->device (drv/get-stream backend) host-buffer 0
                                                (math/device-buffer device-array) 0
                                                (m/ecount host-buffer))
                         [stream device-array])))
                (into {})))
         batch-map-sequence)))


(defn create
  [backend stream-map batch-size]
  (->DatasetBatchingSystem backend (->> stream-map
                                        (map (fn [[k v]]
                                               [k (assoc v
                                                         :batch-buffers
                                                         (create-batch-buffers backend v batch-size))]))
                                        (into {}))))
