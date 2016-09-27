(ns think.compute.compute-utils-test
  (:require [clojure.test :refer :all]
            [think.compute.cpu-device :as cpu-dev]
            [think.compute.device :as dev]
            [think.compute.math :as math]
            [think.compute.compute-utils :as cu]
            [think.compute.datatype :as dtype]))


(deftest compute-map
  (let [data-size 50
        result-seq
        (cu/compute-map (fn []
                          (let [device (cpu-dev/create-device)
                                stream (dev/create-stream device)
                                upload-buffer (dev/allocate-host-buffer device data-size :double)
                                process-buffer (dev/allocate-device-buffer device
                                                                           data-size :double)
                                result-buffer (dev/allocate-device-buffer device data-size
                                                                          :double)

                                ]
                            {:device device :stream stream :io-buf upload-buffer
                             :proc-buf process-buffer
                             :result-buf result-buffer}))
                        (fn [{:keys [device stream io-buf proc-buf]} data]
                          (dtype/copy! data 0 io-buf 0 data-size)
                          (dev/copy-host->device stream io-buf 0
                                                 proc-buf 0 data-size)
                          (math/sum stream 1.0 proc-buf 1.0 proc-buf)
                          (dev/copy-device->host stream proc-buf 0
                                                 io-buf 0 data-size)
                          (dev/wait-for-event (dev/create-event stream))
                          (let [retval (dtype/make-array-of-type :double data-size)]
                            (dtype/copy! io-buf 0 retval 0 data-size)
                            retval))
                        (map #(double-array (repeat data-size %))
                             (range 5)))]
    (is (= (map (fn [idx]
                  (map double (repeat 50 (* 2 idx))))
                (range 5))
           (map seq result-seq)))))
