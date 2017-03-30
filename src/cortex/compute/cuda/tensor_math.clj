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

(defn- operation->cuda
  ([operation]
   [(condp = operation
       :+ (int 0)
       :- (int 1)
       :* (int 2)
       :/ (int 3))])
  ([operation rev-ops?]
   [(operation->cuda operation)
    (int (if rev-ops? 1 0))]))


(defonce cuda_typename_expansion
  [["int8_t" "_b"]
   ["int16_t" "_s"]
   ["int32_t" "_i"]
   ["int64_t" "_l"]
   ["f32_t" "_f"]
   ["f64_t" "_d"]])


(defn- print-datatypes-h-2-dtype-expansion
  []
  (with-out-str
    (println "#define ITERATE_2_DATATYPES\\")
    (doall
     (for [lhs cuda_typename_expansion
           rhs cuda_typename_expansion]
       (println (apply format "  DATATYPE_2_ITERATOR(%s,%s,%s,%s)\\"
                       (flatten [lhs rhs])))))))


(extend-type CudaStream
  tm/TensorMath
  (assign-constant! [stream buffer index-system value n-elems]
    (let [datatype (dtype/get-datatype buffer)
          value (cuda-base/dtype-cast value datatype)
          assign-fn (cuda-base/get-or-create-fn stream :tensor-assign-constant datatype
                                                #(cuda-base/load-all-datatype-function
                                                  "tensor_assign_constant"))
          n-elems (long n-elems)]
      (apply cuda-base/launch-linear-kernel
             (concat [stream assign-fn n-elems 0
                      (cuda-base/->ptr buffer)]
                     (index-system->cuda index-system)
                     [value n-elems]))))
  (assign! [stream
            dest dest-idx-sys
            src src-idx-sys
            n-elems]
    (let [lhs-dtype (dtype/get-datatype dest)
          rhs-dtype (dtype/get-datatype src)
          assign-fn (cuda-base/get-or-create-fn stream :tensor-assign [lhs-dtype rhs-dtype]
                                                #(cuda-base/load-2-datatype-function
                                                  "tensor_assign"))]
      (apply cuda-base/launch-linear-kernel
             (concat [stream assign-fn n-elems 0]
                     [(cuda-base/->ptr dest)]
                     (index-system->cuda dest-idx-sys)
                     [(cuda-base/->ptr src)]
                     (index-system->cuda src-idx-sys)
                     [n-elems]))))

  (binary-accum-constant! [stream
                           dest dest-idx dest-alpha
                           scalar
                           n-elems operation reverse-operands?]
    (let [dest-dtype (dtype/get-datatype dest)
          binop-fn (cuda-base/get-or-create-fn stream :tensor-accum-constant
                                               dest-dtype
                                               #(cuda-base/load-cas-datatype-function
                                                 "tensor_accum_constant"))]
      (apply cuda-base/launch-linear-kernel
             (concat [stream binop-fn n-elems 0]
                     [(cuda-base/->ptr dest)]
                     (index-system->cuda dest-idx)
                     [dest-alpha scalar]
                     (operation->cuda operation reverse-operands?)
                     [n-elems]))))

  (binary-op-constant! [stream
                        dest dest-idx
                        x x-idx x-alpha
                        scalar
                        n-elems operation reverse-operands?]
    (let [dest-dtype (dtype/get-datatype dest)
          binop-fn (cuda-base/get-or-create-fn stream :tensor-binary-op-constant
                                               dest-dtype
                                               #(cuda-base/load-all-datatype-function
                                                 "tensor_binary_op_constant"))]
      (apply cuda-base/launch-linear-kernel
             (concat [stream binop-fn n-elems 0]
                     [(cuda-base/->ptr dest)]
                     (index-system->cuda dest-idx)
                     [(cuda-base/->ptr x)]
                     (index-system->cuda x-idx)
                     [x-alpha scalar]
                     (operation->cuda operation reverse-operands?)
                     [n-elems]))))

  (binary-accum! [stream
                  dest dest-idx dest-alpha
                  y y-idx y-alpha
                  n-elems operation reverse-operands?]
    (let [dest-dtype (dtype/get-datatype dest)
          binop-fn (cuda-base/get-or-create-fn stream :tensor-binary-accum
                                               dest-dtype
                                               #(cuda-base/load-cas-datatype-function
                                                 "tensor_binary_accum"))]
      (apply cuda-base/launch-linear-kernel
             (concat [stream binop-fn n-elems 0]
                     [(cuda-base/->ptr dest)]
                     (index-system->cuda dest-idx)
                     [dest-alpha]
                     [(cuda-base/->ptr y)]
                     (index-system->cuda y-idx)
                     [y-alpha]
                     (operation->cuda operation reverse-operands?)
                     [n-elems]))))

  (binary-op! [stream
               dest dest-idx
               x x-idx x-alpha
               y y-idx y-alpha
               n-elems operation]
    (let [dest-dtype (dtype/get-datatype dest)
          binop-fn (cuda-base/get-or-create-fn stream :tensor-binary-op
                                               dest-dtype
                                               #(cuda-base/load-all-datatype-function
                                                 "tensor_binary_op"))]
      (apply cuda-base/launch-linear-kernel
             (concat [stream binop-fn n-elems 0]
                     [(cuda-base/->ptr dest)]
                     (index-system->cuda dest-idx)
                     [(cuda-base/->ptr x)]
                     (index-system->cuda x-idx)
                     [x-alpha]
                     [(cuda-base/->ptr y)]
                     (index-system->cuda y-idx)
                     [y-alpha]
                     (operation->cuda operation)
                     [n-elems])))))
