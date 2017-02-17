(ns think.compute.compute-utils-test
  (:require [clojure.test :refer :all]
            [think.compute.cpu-driver :as cpu-drv]
            [think.compute.driver :as drv]
            [think.compute.math :as math]
            [think.compute.compute-utils :as cu]
            [think.datatype.core :as dtype]))


(deftest compute-map
  (let [data-size 50
        result-seq
        (cu/compute-map (fn []
                          (let [driver (cpu-drv/create-driver)
                                stream (drv/create-stream driver)
                                upload-buffer (drv/allocate-host-buffer driver data-size :double)
                                process-buffer (drv/allocate-device-buffer driver
                                                                           data-size :double)
                                result-buffer (drv/allocate-device-buffer driver data-size
                                                                          :double)

                                ]
                            {:driver driver :stream stream :io-buf upload-buffer
                             :proc-buf process-buffer
                             :result-buf result-buffer}))
                        (fn [{:keys [driver stream io-buf proc-buf]} data]
                          (dtype/copy! data 0 io-buf 0 data-size)
                          (drv/copy-host->device stream io-buf 0
                                                 proc-buf 0 data-size)
                          (math/sum stream 1.0 proc-buf 1.0 proc-buf)
                          (drv/copy-device->host stream proc-buf 0
                                                 io-buf 0 data-size)
                          (drv/wait-for-event (drv/create-event stream))
                          (let [retval (dtype/make-array-of-type :double data-size)]
                            (dtype/copy! io-buf 0 retval 0 data-size)
                            retval))
                        (map #(double-array (repeat data-size %))
                             (range 5)))]
    (is (= (map (fn [idx]
                  (map double (repeat 50 (* 2 idx))))
                (range 5))
           (map seq result-seq)))))
