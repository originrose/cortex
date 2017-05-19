(ns cortex.verify.tensor
  (:require [cortex.tensor :as ct]
            [cortex.compute.cpu.driver :as cpu-driver]
            [cortex.compute.driver :as drv]
            [clojure.test :refer :all]
            [clojure.core.matrix :as m]))


(defmacro tensor-context
  [driver datatype & body]
  `(drv/with-compute-device
     (drv/default-device ~driver)
     (with-bindings {#'ct/*stream* (drv/create-stream)
                     #'ct/*datatype* ~datatype}
       ~@body)))


(defn assign-constant!
  [driver datatype]
  (tensor-context
   driver datatype
   (let [tensor (ct/->tensor (partition 3 (range 9)))]
     (is (= (ct/ecount tensor) 9))
     (is (m/equals (range 9)
                   (ct/to-double-array tensor)))
     (ct/assign! tensor 1)
     (is (m/equals (repeat 9 1)
                   (ct/to-double-array tensor)))

     (let [rows (ct/rows tensor)
           columns (ct/columns tensor)]
       (doseq [row rows]
         (ct/assign! row 2))
       (is (m/equals (repeat 9 2)
                     (ct/to-double-array tensor)))
       (let [[c1 c2 c3] columns]
         (ct/assign! c1 1)
         (ct/assign! c2 2)
         (ct/assign! c3 3))
       (is (m/equals (flatten (repeat 3 [1 2 3]))
                     (ct/to-double-array tensor)))))))


(defn assign-marshal
  "Assignment must be capable of marshalling data.  This is an somewhat difficult challenge
for the cuda backend."
  [driver datatype]
  (tensor-context
   driver datatype
   (let [tensor (ct/->tensor (partition 3 (range 9)))
         intermediate (ct/new-tensor [3 3] :datatype :int)
         final (ct/new-tensor [3 3] :datatype :double)]
     (ct/assign! intermediate tensor)
     (ct/assign! final intermediate)
     (is (m/equals (range 9)
                   (ct/to-double-array final))))))


(defn binary-constant-op
  [driver datatype]
  (tensor-context
   driver datatype
   (let [tens-a (ct/->tensor (partition 3 (range 9)))
         tens-b (ct/->tensor (repeat 9 1))]

     (when-not (= datatype :byte)
       (ct/binary-op! tens-b 2.0 tens-a 3.0 4.0 :*)
       (is (m/equals (mapv #(* 24 %) (range 9))
                     (ct/to-double-array tens-b))))

     (ct/binary-op! tens-b 2.0 tens-a 3.0 4.0 :+)
     (is (m/equals (mapv #(+ 12 (* 2 %)) (range 9))
                   (ct/to-double-array tens-b)))


     ;;Check reversing operands works.
     (ct/binary-op! tens-b 3.0 4.0 2.0 tens-a :-)
     (is (m/equals (mapv #(- 12 (* 2 %)) (range 9))
                   (ct/to-double-array tens-b)))

     (ct/assign! tens-b 1.0)
     (is (m/equals (repeat 9 1)
                   (ct/to-double-array tens-b)))

     ;;Check accumulation
     (ct/binary-op! tens-b 1.0 tens-b 1.0 1.0 :+)
     (is (m/equals (mapv #(+ 1 %) (repeat 9 1))
                   (ct/to-double-array tens-b))))))


(defn binary-op
  [driver datatype]
  (tensor-context
   driver datatype
   (let [tens-a (ct/->tensor (partition 3 (range 9)))
         tens-b (ct/->tensor (repeat 9 2))
         tens-c (ct/->tensor (repeat 9 10))]
     (ct/binary-op! tens-c 2.0 tens-a 2.0 tens-b :*)
     (is (m/equals (mapv #(* 2 2 2 %) (flatten (partition 3 (range 9))))
                   (ct/to-double-array tens-c)))
     (ct/binary-op! tens-b 1.0 tens-c 2.0 tens-a :-)
     (is (m/equals [0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0]
                   (ct/to-double-array tens-b)))

     ;;A binary accumulation operation where the destination is the same
     ;;as one of the operands.
     (ct/binary-op! tens-c 1.0 tens-c 2.0 tens-a :-)
     (is (m/equals [0.0, 6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0]
                   (ct/to-double-array tens-c)))
     (let [tens-c-small (ct/subvector tens-c 0 :length 3)
           sub-fn (fn [nth-idx]
                    (->> (partition 3 (range 9))
                         (map #(nth % nth-idx))))
           c-data (vec (ct/to-double-array tens-c))]
       (ct/binary-op! tens-c-small 1.0 tens-c-small 2.0 tens-a :-)
       (is (m/equals (mapv (fn [elem-idx]
                             (apply -
                                    (nth c-data elem-idx)
                                    (mapv #(* 2.0 %)
                                          (sub-fn elem-idx))))
                           [0 1 2])
                     (ct/to-double-array tens-c-small)))
       (let [c-data (vec (ct/to-double-array tens-c-small))]
         (ct/binary-op! tens-c-small 2.0 tens-a 1.0 tens-c-small :+)
         (is (m/equals (mapv (fn [elem-idx]
                               (reduce (fn [result a-elem]
                                         (+ a-elem result))
                                       (nth c-data elem-idx)
                                       (map #(* 2.0 %) (sub-fn elem-idx))))
                             [0 1 2])
                       (ct/to-double-array tens-c-small))))))))


(defn gemm
  ;;Test gemm, also test that submatrixes are working and defined correctly.
  [driver datatype]
  (tensor-context
   driver datatype
   (let [tens-a (ct/->tensor (partition 3 (range 9)))
         tens-b (ct/->tensor (partition 3 (repeat 9 2)))
         tens-c (ct/->tensor (partition 3 (repeat 9 10)))]
     (ct/gemm! tens-c false false 1 tens-a tens-b 1)
     (is (m/equals (ct/to-double-array tens-c)
                   [16.0 16.0 16.0 34.0 34.0 34.0 52.0 52.0 52.0]))
     (ct/gemm! tens-c false false 1 tens-a tens-b 0)
     (is (m/equals (ct/to-double-array tens-c)
                   [6.0 6.0 6.0 24.0 24.0 24.0 42.0 42.0 42.0]))
     (ct/gemm! tens-c true false 1 tens-a tens-b 0)
     (is (m/equals (ct/to-double-array tens-c)
                   [18.0, 18.0, 18.0, 24.0, 24.0, 24.0, 30.0, 30.0, 30.0]))
     (let [tens-a (ct/submatrix tens-a 0 2 0 2)
           tens-b (ct/submatrix tens-b 0 2 0 2)
           tens-c-sub (ct/submatrix tens-c 0 2 0 2)]
       (ct/gemm! tens-c-sub false false 1 tens-a tens-b 0)
       (is (m/equals (ct/to-double-array tens-c-sub)
                     [2 2 14 14]))
       (is (m/equals (ct/to-double-array tens-c)
                     [2.0, 2.0, 18.0, 14.0, 14.0, 24.0, 30.0, 30.0, 30.0])))
     (let [tens-a (ct/submatrix tens-a 1 2 1 2)
           tens-b (ct/submatrix tens-b 0 2 0 2)
           tens-c-sub (ct/submatrix tens-c 1 2 1 2)]
       (ct/gemm! tens-c-sub false false 1 tens-a tens-b 1)
       (is (m/equals (ct/to-double-array tens-c-sub)
                     [32 42 60 60]))
       (is (m/equals (ct/to-double-array tens-c)
                     [2.0, 2.0, 18.0, 14.0, 32.0, 42.0, 30.0, 60.0, 60.0]))))))
