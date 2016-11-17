(ns think.compute.verify.batching-system
    (:require [clojure.test :refer :all]
            [think.compute.batching-system :as batching-system]
            [cortex.dataset :as ds]
            [think.datatype.core :as dtype]
            [think.compute.driver :as drv]
            [think.compute.math :as math]))


(defn generate-buffer-seq
  [data-size index-seq]
  (let [output-buffer (double-array data-size)]
    (map (fn [idx]
           (dtype/copy-raw->item! (range (* data-size idx) (* data-size (+ idx 1)))
                                  output-buffer 0)
           output-buffer)
         index-seq)))


(defrecord GenerativeDataset [^long input-size ^long output-size ^long num-indexes]
  ds/PDataset
  (shapes [ds] {:input input-size
                :output output-size})
  (get-batches [ds batch-size batch-type elem-names]
    (let [indexes (vec (range num-indexes))
          batches (partition batch-size indexes)]
      (map (fn [batch-indexes]
             (mapv (fn [data-name]
                     (condp = data-name
                       :input (generate-buffer-seq input-size batch-indexes)
                       :output (generate-buffer-seq output-size batch-indexes)))
                   elem-names))
           batches))))


(defn copy-device-buffer-to-output-ary
  [device-buffer batch-idx output-ary driver stream]
  (let [double-data (math/to-double-array driver stream device-buffer)
        num-items (alength double-data)
        offset (* num-items batch-idx)]
    (dtype/copy! double-data 0 output-ary offset num-items)))


(defn full-batching-system-test
  "Run through the data anad make sure we are getting the numbers back we expect to get back."
  [driver datatype]
  (let [stream (drv/create-stream driver)
        input-size 100
        output-size 20
        num-elems 200
        batch-size 10
        num-batches (quot num-elems batch-size)
        dataset (->GenerativeDataset input-size output-size num-elems)
        input-record (double-array (* input-size num-elems))
        output-record (double-array (* output-size num-elems))
        system (-> (batching-system/create-dataset-batching-system [:input] [:output] batch-size dataset driver stream datatype)
                   (batching-system/setup))]
    (doseq [[batch-idx batch-data] (map-indexed vector (batching-system/get-batches system :training true))]
      (let [{:keys [input-buffers output-buffers]} batch-data]
        (copy-device-buffer-to-output-ary (input-buffers 0) batch-idx input-record driver stream)
        (copy-device-buffer-to-output-ary (output-buffers 0) batch-idx output-record driver stream)))

    (is (= (vec input-record) (mapv double (range (* input-size num-elems)))))
    (is (= (vec output-record) (mapv double (range (* output-size num-elems)))))))
