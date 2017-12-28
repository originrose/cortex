(ns cortex.compute.verify.driver
  (:require [clojure.test :refer :all]
            [cortex.compute.driver :as drv]
            [cortex.compute.math :as math]
            [think.datatype.core :as dtype]
            [think.datatype.base :as dtype-base]
            [clojure.core.matrix :as m]))



(defn simple-stream
  [driver datatype]
  (drv/with-compute-device
    (drv/default-device driver)
    (let [stream (drv/create-stream)
          buf-a (drv/allocate-host-buffer driver 10 datatype)
          output-buf-a (drv/allocate-host-buffer driver 10 datatype)
          buf-b (drv/allocate-device-buffer 10 datatype)
          input-data (dtype/make-array-of-type datatype (range 10))
          output-data (dtype/make-array-of-type datatype 10)]
      (dtype/copy! input-data 0 buf-a 0 10)
      (dtype-base/set-value! buf-a 0 100.0)
      (dtype/copy! buf-a 0 output-data 0 10)
      (drv/copy-host->device stream buf-a 0 buf-b 0 10)
      (drv/memset stream buf-b 5 20.0 5)
      (drv/copy-device->host stream buf-b 0 output-buf-a 0 10)
      (drv/sync-stream stream)
      (dtype/copy! output-buf-a 0 output-data 0 10)
      (is (= [100.0 1.0 2.0 3.0 4.0 20.0 20.0 20.0 20.0 20.0]
             (mapv double output-data))))))
