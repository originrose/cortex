(ns cortex.tensor.allocator
  (:require [cortex.tensor :as ct]
            [cortex.tensor.dimensions :as dims]
            [cortex.compute.driver :as compute-drv]
            [clojure.core.matrix :as m]
            [think.datatype.core :as dtype]))


(defprotocol PAllocator
  "Allocation protocol to allow different allocation strategies"
  (->tensor-impl [this name data args]
    "Allocate a tensor.  IF allocated, copy existing data into existing tensor.")

  (->const-tensor-impl [this name data args]
    "Allocate a tensor.  If allocated, do nothing and return tensor")

  (new-tensor-impl [this name shape args]
    "Create a new tensor.  If allocated, initialize to 0")

  (new-uninitialized-tensor-impl [this name shape args]
    "Create a new tensor.  If allocated, do nothing"))


(defn- args->map
  [args]
  (->> (partition 2 args)
       (map vec)
       (into {})))


(defn- get-or-create-item
  [map-atom key value-fn]
  (let [retval (get @map-atom key)]
    (when-not retval
      (swap! map-atom #(assoc % key (value-fn))))
    (get @map-atom key)))


(defrecord BaseTensorAllocator [allocated-data]
  PAllocator
  (->tensor-impl
    [this name data args]
    (let [{:keys [datatype]
           :or {datatype ct/*datatype*}} (args->map args)
          stream (ct/check-stream)
          data-shape (m/shape data)
          n-elems (long (apply * 1 data-shape))
          dimensions (dims/dimensions data-shape)
          device (compute-drv/get-device stream)
          {:keys [host-buffer dev-buffer tensor]}
          (get-or-create-item
           allocated-data name
           (fn []
             (let [host-buffer (compute-drv/allocate-host-buffer
                                (compute-drv/get-driver device)
                                n-elems datatype)
                   dev-buffer (compute-drv/allocate-device-buffer n-elems datatype
                                                                     :device device)
                   tensor (ct/construct-tensor device dimensions dev-buffer)]
              {:host-buffer host-buffer
               :dev-buffer dev-buffer
               :tensor tensor})))]

      (dtype/copy-raw->item! data host-buffer 0)
      (compute-drv/copy-host->device stream host-buffer 0 dev-buffer 0 n-elems)
      tensor))

  (->const-tensor-impl
    [this name data args]
    (let [{:keys [datatype]
           :or {datatype ct/*datatype*}} (args->map args)
          stream (ct/check-stream)
          device (compute-drv/get-device stream)]
      (if-let [retval (get @allocated-data name)]
        (get retval :tensor)
        (->tensor-impl this name data args))))

  (new-tensor-impl
    [this name shape args]
    (if-let [retval (get @allocated-data name)]
      (do
        (m/assign! retval 0)
        retval)
      (let [retval (apply ct/new-tensor shape args)]
        (swap! allocated-data assoc name retval)
        retval)))

  (new-uninitialized-tensor-impl
    [this name shape args]
    (if-let [retval (get @allocated-data name)]
      retval
      (new-tensor-impl this name shape args))))


(defn atom-allocator [] (->BaseTensorAllocator (atom {})))


(defrecord PassthroughAllocator []
  PAllocator
  (->tensor-impl [this name data args]
    (apply ct/->tensor data args))

  (->const-tensor-impl [this name data args]
    (apply ct/->tensor data args))

  (new-tensor-impl [this name shape args]
    (apply ct/new-tensor shape args))

  (new-uninitialized-tensor-impl [this name shape args]
    (apply ct/new-tensor shape args)))


(defn passthrough-allocator [] (->PassthroughAllocator))


(defonce ^:dynamic *allocator* nil)


(defmacro with-allocator
  [allocator & body]
  `(with-bindings [#'*allocator* ~allocator]
     ~@body))


(defn get-allocator
  []
  *allocator*)


(defn check-allocator
  []
  (when-not *allocator*
    (throw (ex-info "Allocation attempted with no allocator defined.")))
  *allocator*)


(defn ->tensor
  [name data & args]
  (->tensor-impl (check-allocator) name data args))


(defn ->const-tensor
  [name data & args]
  (->const-tensor-impl (check-allocator) name data args))


(defn new-tensor
  [name shape & args]
  (new-tensor-impl (check-allocator) name shape args))


(defn new-uninitialized-tensor
  [name shape & args]
  (new-uninitialized-tensor-impl (check-allocator) name shape args))
