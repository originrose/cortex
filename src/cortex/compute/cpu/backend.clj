(ns cortex.compute.cpu.backend
  (:require
    [clojure.core.matrix :as m]
    [clojure.core.matrix.protocols :as mp]
    [clojure.core.matrix.macros :refer [c-for]]
    [think.datatype.core :refer [v-aget v-aset v-alength] :as dtype]
    [think.datatype.base :as dtype-base]
    [cortex.graph :as graph]
    [cortex.nn.layers :as layers]
    [cortex.nn.impl :as impl]
    [cortex.compute.driver :as drv]
    [cortex.compute.math :as math]
    [cortex.compute.cpu.driver :as cpu-drv]
    [cortex.compute.nn.layers :as compute-layers]
    [cortex.compute.nn.protocols :as compute-protocols]
    [cortex.compute.nn.backend :as nn-backend]
    [think.resource.core :as resource]
    [cortex.compute.cpu.tensor-math]
    [think.parallel.core :as parallel])
  (:import
    [java.util Arrays]
    [java.util.concurrent ForkJoinPool Callable Future]
    [java.nio DoubleBuffer FloatBuffer]
    [think.datatype ArrayView DoubleArrayView FloatArrayView]
    [cortex.compute.cpu.driver CPUDriver CPUStream]
    [cortex.compute.math DeviceArray Tensor]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defrecord CPUBackend [type device stream datatype resource-context]
  dtype-base/PDatatype
  (get-datatype [backend] (.datatype backend))
  drv/PDriverProvider
  (get-driver [backend] (drv/get-driver (.device backend)))
  drv/PDeviceProvider
  (get-device [backend] (.device backend))
  drv/PStreamProvider
  (get-stream [backend] (.stream backend))
  resource/PResource
  (release-resource [backend]
    (drv/unsafe-with-compute-device
     (.device backend)
     (resource/release-resource-context @(get backend :resource-context)))))


(defn backend
  [& {:keys [datatype driver device stream]}]
  (let [datatype (or datatype :double)
        driver (or driver (cpu-drv/driver))
        device (or device (drv/default-device driver))]
    (drv/unsafe-with-compute-device
     device
     (let [stream (or stream (drv/create-stream))
           [backend res-ctx]
           (resource/return-resource-context
            (->CPUBackend :cpu device stream datatype (atom nil)))]
       (reset! (get backend :resource-context) res-ctx)
       (resource/track backend)))))


(defn device-array->view
  [dev-ary & [size]]
  (if size
    (dtype/->view (math/device-buffer dev-ary) 0 size)
    (dtype/->view (math/device-buffer dev-ary))))
