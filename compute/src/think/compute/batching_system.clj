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
  ;;Returns a sequence where each item of the sequence contains:
  ;;{:input-buffers - vector of buffers used for input
  ;; :output-buffers - vector of buffers used for output
  ;;}
  ;;There is an option to skip the upload steps to the output buffers which
  ;;aren't necessary if you aren't doing gradient descent (e.g. any inference).
  (get-batches [bs batch-map-sequence upload-output-buffers?]))


(defrecord DatasetBatchingSystem [backend stream->batch-info-map]
  PBatchingSystem
  (get-batches [bs batch-map-sequence upload-output-buffers?]
    (let [necessary-buffers (if upload-output-buffers?
                              (seq stream->batch-info-map)
                              (->> stream->batch-info-map
                                   (filter (fn [[k v]]
                                          (contains? (get v :direction) :input)))))
          necessary-keys (mapv first necessary-buffers)]
     (map (fn [batch-map]
            (when-not (every? #(contains? batch-map %) necessary-keys)
              (throw (ex-info "Network batching Missing streams:"
                              {:dataset-streams (keys batch-map)
                               :network-streams necessary-keys})))
            (->> necessary-buffers
                 (map (fn [[stream {:keys [batch-buffers size]}]]
                        (let [{:keys [device-array host-buffer]} batch-buffers]
                          (dtype/copy-raw->item! (get batch-map stream) host-buffer 0)
                          (drv/copy-host->device (drv/get-stream backend) host-buffer 0
                                                 (math/device-buffer device-array) 0
                                                 (m/ecount host-buffer))
                          [stream device-array])))
                 (into {})))
          batch-map-sequence))))


(defn- create-batch-buffers
  [backend size batch-size]
  (let [driver (drv/get-driver backend)
        datatype (dtype/get-datatype backend)]
   {:device-array (math/new-array driver
                                  (drv/get-stream backend)
                                  datatype
                                  [size]
                                  batch-size)
    :host-buffer (drv/allocate-host-buffer driver (* size batch-size) datatype)}))


(defn create
  [backend stream-map batch-size]
  (->DatasetBatchingSystem backend (->> stream-map
                                        (map (fn [[k v]]
                                               (let [item-size (if (map? v)
                                                                 (get v :shape)
                                                                 (long v))
                                                     v (if (map? v)
                                                         v
                                                         {})]
                                                 [k (assoc v
                                                           :size item-size
                                                           :batch-buffers
                                                           (create-batch-buffers backend
                                                                                 item-size
                                                                                 batch-size))])))
                                        (into {}))))
