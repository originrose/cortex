(ns cortex.optimize.sgd
  (:require
   [cortex.optimize :refer [PGradientOptimizer create-optimizer]]
   [cortex.compute.math :as math]
   [think.datatype.core :as dtype]
   [cortex.util :as util]
   [cortex.graph :as graph]
   [cortex.compute.nn.backend :as compute-backend]
   [cortex.tensor :as ct]
   [cortex.tensor.allocator :as ct-alloc]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn tensor-sgd
  [allocator learning-rate momentum momentum-vals
   gradient-alpha offset gradient parameters]
  (ct/with-stream (compute-backend/get-stream)
    (ct/with-datatype (dtype/get-datatype gradient)
     (ct-alloc/with-allocator allocator
       (let [momentum (double momentum)
             gradient (ct/as-vector (math/array->cortex-tensor gradient))
             parameters (ct/as-vector (math/array->cortex-tensor parameters))
             momentum-vals (-> (ct/as-vector (math/array->cortex-tensor momentum-vals))
                               (ct/subvector offset :length (ct/ecount gradient)))
             gradient-temp (ct-alloc/new-resizeable-uninitialized-tensor :sgd-temp [(ct/ecount gradient)])
             ;;Use gradient-temp for dxm
             dxm (ct/unary-op! gradient-temp momentum momentum-vals :noop)
             ;;update momentum with gradient
             new-momentum (ct/binary-op! momentum-vals
                                         1.0 dxm
                                         (* (double learning-rate)
                                            (double gradient-alpha))
                                         gradient
                                         :+)
             ;;create adjusted dx
             dx (ct/binary-op! gradient-temp 1.0 dxm (+ 1 momentum) momentum-vals :-)
             ;;Sum into parameters
             new-parameters (ct/binary-op! parameters 1.0 parameters 1.0 dx :+)])))))


(defrecord SGD [backend optimizer alloc]
  PGradientOptimizer
  (batch-update [this optimizer-parameters]
    this)
  (compute-parameters! [this {:keys [momentum-vals] :as params}
                        gradient-alpha offset gradient parameters]
    (let [{:keys [learning-rate momentum]} optimizer]
      (tensor-sgd alloc learning-rate momentum momentum-vals
                  gradient-alpha offset gradient parameters))))


(defn- setup-optimizer
  [backend optimizer]
  (->SGD backend optimizer (ct-alloc/atom-allocator)))


(defmethod create-optimizer [:cpu :sgd]
  [backend optimizer]
  (setup-optimizer backend optimizer))

(defmethod create-optimizer [:cuda :sgd]
  [backend optimizer]
  (setup-optimizer backend optimizer))

(defmethod graph/get-node-metadata :sgd
  [desc]
  {:arguments
   {:momentum-vals {:initialization {:type :constant :value 0}
                    :type :parameter}}
   :passes #{:training}})

(defn sgd
  [& args]
  (util/merge-args
   {:type :sgd
    :learning-rate 0.001
    :momentum 0.9}
   args))
