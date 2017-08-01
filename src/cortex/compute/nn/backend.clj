(ns cortex.compute.nn.backend
  "Neural network backends provide the driver-specific computations that cannot be represented
  with the generalized math layer provided in math.clj or where cudnn provides a specific optimized
  implementation.
  A backend is expected to have access to:
  1.  A specific driver.
  2.  A stream of execution.
  3.  A datatype used to specify what the backing data should be.

  It is also expected to be capable of providing backend specific implementations for various layer types.
  There are a set of functions that correspond to some specific math functions but take a backend instead
  of a driver and stream to streamline creating data for a given backend."

  (:require [cortex.compute.math :as math]
            [cortex.compute.driver :as drv]
            [think.datatype.core :as dtype]
            [cortex.tensor :as tensor]))


(def ^:dynamic *current-backend-stream* nil)


(defmacro with-backend
  [backend & body]
  `(let [backend# ~backend]
     (-> (drv/with-compute-device (drv/get-device backend#)
           (with-bindings {#'*current-backend-stream* (drv/get-stream backend#)}
             ~@body))
         ;;let an enclosing resource context deal with the resources created while the
         ;;compute device was active.
         first)))


(defmacro with-stream
  [stream & body]
  `(with-bindings {#'*current-backend-stream* ~stream}
     ~@body))


(defn get-stream
  []
  (when-not *current-backend-stream*
    (throw (ex-info "Backend stream is nil.  Use backend/with-backend to set." {})))
  *current-backend-stream*)


(defprotocol PLayerCreation
  "For layers completely implemented in the backend we allow the backend to create
some specific data from a description.  Most layers need to implement
computelayer/forward,backward."
  (create [backend layer batch-size]))

(defprotocol PDropout
  ;;Flat distribution -> scaled 1 or 0 multiplicative buffer.
  (prepare-bernoulli-dropout! [backend probability rand-buffer mult-buffer])
  ;;Gaussian distribution copied to mult buffer.
  (prepare-gaussian-dropout! [backend rand-buffer mult-buffer]))

(defn array
  ([backend data items-per-batch]
   (math/array (get-stream) (dtype/get-datatype backend)
               data items-per-batch))
  ([backend data]
   (array backend data 1)))

(defn new-array
  ([backend shape items-per-batch]
   (math/new-array (get-stream) (dtype/get-datatype backend)
                   shape items-per-batch))
  ([backend shape]
   (new-array backend shape 1)))

(defn allocate-ones [backend elem-count]
  (math/allocate-ones (get-stream)
                      (dtype/get-datatype backend) elem-count))

(defn allocate-rand-buffer
  [backend elem-count]
  (math/allocate-rand-buffer elem-count))

(defn assign!
  [backend dest src]
  (math/assign! (get-stream) dest src))

(defn to-core-matrix
  [backend ary & opts]
  (apply math/to-core-matrix (get-stream) ary (math/shape ary) opts))

(defn device-array->array
  [backend datatype device-ary]
  (math/device-array->array (get-stream) datatype device-ary))

(defn to-double-array
  [backend ary]
  (device-array->array backend :double ary))


(defn zero-many!
  [backend dev-array-seq]
  (doseq [ary dev-array-seq]
    (drv/memset (get-stream) (math/device-buffer ary) 0 0 (math/ecount ary))))


(defn- ->ct-tensor
  [ary]
  (when ary
    (math/array->cortex-tensor ary)))


(defn biased-multiply!
  [backend input weights bias output]
  (let [input (->simple-batch-tensor input)
        weights (->ct-tensor weights)
        bias (->ct-tensor bias)
        output (->ct-tensor output)]
    (tensor/with-stream (get-stream)
      (tensor/binary-op! output 1.0 bias 0.0 output :+)
      (tensor/gemm! output false true
                    1.0 (tensor/as-batch-matrix input) weights
                    1.0))))


(defn biased-multiply-backward!
  [backend input weights bias output
   input-gradient weight-gradient bias-gradient output-gradient]
  (let [input (->ct-tensor input)
        weights (->ct-tensor weights)
        bias (->ct-tensor bias)
        output (->ct-tensor output)
        input-gradient (->ct-tensor input-gradient)
        weight-gradient (->ct-tensor weight-gradient)
        bias-gradient (->ct-tensor bias-gradient)
        output-gradient (->ct-tensor output-gradient)]
    (tensor/with-stream (get-stream)
     (when bias-gradient
       (tensor/binary-op! bias-gradient 1.0 output-gradient 1.0 bias-gradient :+))
      (when input-gradient
        (tensor/gemm! (tensor/as-batch-matrix input-gradient)
                      false false 1.0
                      (tensor/as-batch-matrix output-gradient) weights
                      0.0))
      (when weight-gradient
        (tensor/gemm! weight-gradient
                      true false 1.0
                      (tensor/as-batch-matrix output-gradient)
                      (tensor/as-batch-matrix input)
                      1.0)))))
