(ns cortex.compute.cuda.tensor-math
  (:require [cortex.compute.cuda.base :as cuda-base]
            [think.datatype.core :as dtype]
            [cortex.tensor.math :as tm]
            [cortex.tensor.index-system :as is])
  (:import [cortex.compute.cuda.base CudaStream]
           [org.bytedeco.javacpp Pointer IntPointer]))

(defn- strategy-type->int
  ^Integer [index-system]
  (int
   (condp = (get-in index-system [:strategy :type])
     :constant 0
     :monotonically-increasing 1
     :monotonically-decreasing 2
     :indexed 3)))

(defn- strategy->c-or-len
  ^Integer [index-system]
  (let [strategy (get index-system :strategy)]
    (int
     (condp = (get strategy :type)
       :constant (get strategy :constant)
       :monotonically-increasing (get strategy :length)
       :monotonically-decreasing (get strategy :length)
       :indexed (dtype/ecount (get strategy :indexes))))))


(defn- strategy->idx-ptr
  ^IntPointer [index-system]
  (if (= :indexed (get-in index-system [:strategy :type]))
    (cuda-base/->ptr (get-in index-system [:strategy :indexes]))
    (IntPointer.)))


(defn- index-system->cuda
  [index-system]
  [(strategy-type->int index-system)
   (strategy->c-or-len index-system)
   (strategy->idx-ptr index-system)
   (int (or (get index-system :num-columns) 1))
   (int (or (get index-system :column-stride) 1))])


(extend-type CudaStream
  tm/TensorMath
  (assign-constant! [stream buffer index-system value n-elems]
    (let [datatype (dtype/get-datatype buffer)
          value (cuda-base/dtype-cast value datatype)
          assign-fn (cuda-base/get-or-create-fn stream :assign-constant datatype
                                                #(cuda-base/load-all-datatype-function
                                                  "tensor_assign_constant"))
          n-elems (long n-elems)]
      (apply cuda-base/launch-linear-kernel
             (concat [stream assign-fn n-elems 0
                      (cuda-base/->ptr buffer)]
                     (index-system->cuda index-system)
                     [value n-elems]))))
  (assign! [stream
            dest dest-idx-sys dest-ecount
            src src-idx-sys src-ecount]
    (throw (ex-info "Unimplemented" {}))))
