(ns cortex.tensor.allocator
  (:require [cortex.tensor :as ct]
            [cortex.tensor.dimensions :as dims]
            [cortex.compute.driver :as compute-drv]
            [clojure.core.matrix :as m]
            [think.datatype.core :as dtype]
            [think.resource.core :as resource]))


(defprotocol PAllocator
  "Allocation protocol to allow different allocation strategies"
  (->tensor-impl [this name data args]
    "Allocate a tensor.  IF allocated, copy existing data into existing tensor.")

  (->const-tensor-impl [this name data args]
    "Allocate a tensor.  If allocated, do nothing and return tensor.")

  (new-tensor-impl [this name shape args]
    "Create a new tensor.  If allocated, initialize to 0")

  (new-uninitialized-tensor-impl [this name shape args]
    "Create a new tensor.  If allocated, do nothing")

  (new-resizeable-uninitialized-impl [this name shape args]
    "Create a new tensor, possibly making the existing one large.  Return an uninitialized
tensor of exactly the requested shape.")

  (allocation-count-impl [this]))


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


(defn- ensure-shape!
  [tensor new-shape]
  (when-not (= new-shape (ct/shape tensor))
    (throw (ex-info "Shape mismatch between allocated tensor and requested tensor"
                    {:existing-shape (ct/shape tensor)
                     :incoming-shape new-shape})))
  tensor)


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
      (ensure-shape! tensor data-shape)
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
        (ensure-shape! (get retval :tensor) (m/shape data))
        (->tensor-impl this name data args))))

  (new-tensor-impl
    [this name shape args]
    (if-let [retval (get-in @allocated-data [name :tensor])]
      (-> (ensure-shape! retval shape)
          (ct/assign! 0))
      (let [retval (apply ct/new-tensor shape args)]
        (swap! allocated-data assoc name {:tensor retval})
        retval)))

  (new-uninitialized-tensor-impl
    [this name shape args]
    (if-let [retval (get @allocated-data name)]
      (ensure-shape! (:tensor retval) shape)
      (new-tensor-impl this name shape args)))

  (new-resizeable-uninitialized-impl [this name shape args]
    (let [shape-ecount (long (apply * 1 shape))
          retval (get-in @allocated-data [name :tensor])]
      (if retval
        (let [existing-buffer (ct/tensor->buffer retval)
              retval-buffer-ecount (long (m/ecount existing-buffer))
              new-buffer (if (< retval-buffer-ecount shape-ecount)
                             (do
                               (compute-drv/sync-stream ct/*stream*)
                               (resource/release existing-buffer)
                               (compute-drv/allocate-device-buffer shape-ecount
                                                                   (dtype/get-datatype existing-buffer)))
                             existing-buffer)]
          (swap! allocated-data assoc-in [name :tensor :buffer] new-buffer)
          (ct/->Tensor (:device retval) (dims/dimensions shape) new-buffer))
        (new-uninitialized-tensor-impl this name shape args))))

  (allocation-count-impl
    [this]
    (m/esum (mapv (comp m/ecount :tensor second) @allocated-data))))


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
    (apply ct/new-tensor shape args))

  (new-resizeable-uninitialized-impl [this name shape args]
    (apply ct/new-tensor shape args))

  (allocation-count-impl [this]
    0))


(defn passthrough-allocator [] (->PassthroughAllocator))


(defonce ^:dynamic *allocator* nil)


(defmacro with-allocator
  [allocator & body]
  `(with-bindings {#'*allocator* ~allocator}
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


(defn new-resizeable-uninitialized-tensor
  [name shape & args]
  (new-resizeable-uninitialized-impl (check-allocator) name shape args))


(defn allocation-count
  ^long []
  (allocation-count-impl (check-allocator)))
