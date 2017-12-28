(ns cortex.verify.tensor.operations
  (:require [clojure.core.matrix :as m]
            [clojure.test :refer :all]
            [cortex.compute.driver :as drv]
            [cortex.tensor :as ct]
            [cortex.tensor.operations :as tops]
            [think.datatype.core :as dtype]))

(defmacro tensor-context
  [driver datatype & body]
  `(drv/with-compute-device
     (drv/default-device ~driver)
     (with-bindings {#'ct/*stream* (drv/create-stream)
                     #'ct/*datatype* ~datatype}
       ~@body)))

(defn unary-operation [driver datatype op-fn arg src-data compare-result]
  (tensor-context
   driver datatype
   (let [tens-a (ct/->tensor src-data)
         tens-b (tops/new-tensor tens-a)
         result (if arg
                  (op-fn tens-b tens-a arg)
                  (op-fn tens-b tens-a))]
     (is (m/equals compare-result
                   (ct/to-double-array
                    result)
                   1e-4))
     (let [result2 (if arg
                     (op-fn tens-a arg)
                     (op-fn tens-a))]
       (is (m/equals compare-result
                     (ct/to-double-array
                      result)
                     1e-4))))))

(defn max-operation
  [driver datatype]
  (unary-operation driver
                   datatype
                   tops/max
                   2
                   [0 1 2 3 4]
                   [2 2 2 3 4]))

(defn min-operation
  [driver datatype]
  (unary-operation driver
                   datatype
                   tops/min
                   2
                   [0 1 2 3 4]
                   [0 1 2 2 2]))

(defn ceil-operation
  [driver datatype]
  (let [data [0.5 1.5 2.5 3.5]]
    (unary-operation driver
                     datatype
                     tops/ceil
                     nil
                     data
                     (mapv #(Math/ceil (drv/dtype-cast % datatype)) data))))

(defn floor-operation
  [driver datatype]
  (let [data [0.5 1.5 2.5 3.5]]
    (unary-operation driver
                     datatype
                     tops/floor
                     nil
                     data
                     (mapv #(Math/floor (drv/dtype-cast % datatype)) data))))

(defn logistic-operation
  [driver datatype]
  (unary-operation driver
                   datatype
                   tops/logistic
                   nil
                   (range -4 8)
                   [0.01798620996209156, 0.04742587317756678, 0.11920292202211755,
                    0.2689414213699951, 0.5, 0.7310585786300049, 0.8807970779778823,
                    0.9525741268224334, 0.9820137900379085, 0.9933071490757153,
                    0.9975273768433653, 0.9990889488055994]))

(defn tanh-operation
  [driver datatype]
  (unary-operation driver
                   datatype
                   tops/tanh
                   nil
                   (range -4 8)
                   [-0.999329299739067, -0.9950547536867305, -0.9640275800758169,
                    -0.7615941559557649, 0.0, 0.7615941559557649, 0.9640275800758169,
                    0.9950547536867305, 0.999329299739067, 0.9999092042625951,
                    0.9999877116507956, 0.9999983369439447]))

(defn exp-operation
  [driver datatype]
  (let [data [0 1 2 3 4]]
   (unary-operation driver
                    datatype
                    tops/exp
                    nil
                    data
                    (mapv #(drv/dtype-cast (Math/exp (double %)) datatype) data))))

(defn binary-operation [driver datatype op-fn src-data-a src-data-b compare-result]
  (tensor-context
   driver datatype
   (let [tens-a (ct/->tensor src-data-a)
         tens-b (ct/->tensor src-data-b)
         tens-c (tops/new-tensor tens-b)
         result (op-fn tens-c tens-a tens-b)]
     (is (m/equals compare-result
                   (ct/to-double-array result)))
     (let [result2 (op-fn tens-a tens-b)]
       (is (m/equals compare-result
                     (ct/to-double-array
                      result2)))))))

(defn multiply-operation
  [driver datatype]
  (let [x1 [-1 0 1 2 3 4]
        x2 [ 2 3 4 5 6 7]]
   (binary-operation driver
                     datatype
                     tops/*
                     x1
                     x2
                     (mapv #(drv/dtype-cast (* (double %1) (double %2)) datatype) x1 x2))))

(defn add-operation
  [driver datatype]
  (let [x1 [-1 0 1 2 3 4]
        x2 [ 2 3 4 5 6 7]]
    (binary-operation driver
                      datatype
                      tops/+
                      x1
                      x2
                      (mapv #(drv/dtype-cast (+ %1 %2) datatype) x1 x2))))

(defn subtract-operation
  [driver datatype]
  (let [x1 [1 2 3 4 5 6]
        x2 [0 1 2 1 0 1]]
    (binary-operation driver
                      datatype
                      tops/-
                      x1
                      x2
                      (mapv #(drv/dtype-cast (- %1 %2) datatype) x1 x2))))

(defn >-operation
  [driver datatype]
  (binary-operation driver
                    datatype
                    tops/>
                    [-1 0 1 2 3 4]
                    [ 1 1 1 1 1 1]
                    [ 0 0 0 1 1 1]))

(defn >-operation
  [driver datatype]
  (binary-operation driver
                    datatype
                    tops/>=
                    [-1 0 1 2 3 4]
                    [ 1 1 1 1 1 1]
                    [ 0 0 1 1 1 1]))


(defn <-operation
  [driver datatype]
  (binary-operation driver
                    datatype
                    tops/>
                    [-1 0 1 2 3 4]
                    [ 1 1 1 1 1 1]
                    [ 1 1 0 0 0 0]))

(defn <=-operation
  [driver datatype]
  (binary-operation driver
                    datatype
                    tops/>=
                    [-1 0 1 2 3 4]
                    [ 1 1 1 1 1 1]
                    [ 1 1 1 0 0 0]))

(defn bit-and-operation
  [driver datatype]
  (binary-operation driver
                    datatype
                    tops/bit-and
                    [1 0 0 1]
                    [1 1 1 1]
                    [1 0 0 1]))

(defn bit-xor-operation
  [driver datatype]
  (binary-operation driver
                    datatype
                    tops/bit-xor
                    [1 0 0 1]
                    [1 1 1 1]
                    [0 1 1 0]))

(defn where-operation [driver datatype]
  (tensor-context
   driver datatype
   (let [test-ten (ct/->tensor [1 0 0 1 0])
         then-ten (ct/->tensor [1 2 3 4 5])
         else-ten (ct/->tensor [6 7 8 9 10])
         output-ten (tops/new-tensor else-ten)
         result (tops/where output-ten test-ten then-ten else-ten)]
     (is (m/equals [1.0 7.0 8.0 4.0 10.0]
                   (ct/to-double-array result))))))

(defn new-tensor-operation [driver datatype]
  (tensor-context
   driver datatype
   (let [test-ten (ct/->tensor [1 0 0 1 0])
         new-ten (tops/new-tensor test-ten)]
     (is (= (m/shape test-ten) (m/shape new-ten)))
     (is (= (dtype/get-datatype test-ten) (dtype/get-datatype new-ten)))
     (is (m/equals [0.0 0.0 0.0 0.0 0.0]
                   (ct/to-double-array new-ten))))))
