(ns cortex.verify.tensor
  (:require [cortex.tensor :as ct]
            [cortex.compute.driver :as drv]
            [clojure.test :refer :all]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.stats :as stats]
            [cortex.util :as util]
            [cortex.tensor :as tensor]))


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


(defn unary-op
  [driver datatype]
  (tensor-context driver datatype
   (let [tens-a (ct/->tensor (partition 3 (range 9)))
         tens-b (ct/->tensor (partition 3 (repeat 9 1)))]
     (ct/unary-op! tens-b 2.5 tens-a :ceil)
     (is (m/equals (mapv #(Math/ceil (* ^double % (drv/dtype-cast 2.5 datatype))) (range 9))
                   (ct/to-double-array tens-b)))
     (ct/unary-op! tens-b 1.0 tens-b :-)
     (is (m/equals (mapv #(- (Math/ceil (* ^double % (drv/dtype-cast 2.5 datatype)))) (range 9))
                   (ct/to-double-array tens-b)))

     (let [src-data [0 1 2 3 4]
           tens-src (ct/->tensor src-data)
           tens-b (ct/->tensor src-data)]
       (ct/unary-op! tens-b 1.0 tens-src :exp)
       (is (m/equals (mapv #(drv/dtype-cast (Math/exp (double %)) datatype) src-data)
                     (ct/to-double-array tens-b)))
       (ct/unary-op! tens-b 1.0 tens-src :sqrt)
       (is (m/equals (mapv #(drv/dtype-cast (Math/sqrt (double %)) datatype) src-data)
                     (ct/to-double-array tens-b)))))))


(defn channel-op
  [driver datatype]
  (tensor-context driver datatype
   (let [tens-a (ct/->tensor (partition 3 (partition 3 (range 18))))
         tens-chan (ct/in-place-reshape (ct/->tensor (range 3)) [3 1])
         tens-result (ct/new-tensor [2 3 3])]

     (ct/binary-op! tens-result 1.0 tens-a 1.0 tens-chan :*)
     (is (m/equals [0.0, 0.0, 0.0, 3.0, 4.0, 5.0,
                    12.0, 14.0, 16.0, 0.0, 0.0, 0.0, 12.0,
                    13.0, 14.0, 30.0, 32.0, 34.0]
                   (ct/to-double-array tens-result))))))


(defn binary-constant-op
  [driver datatype]
  (tensor-context
   driver datatype
   (let [tens-a (ct/->tensor (partition 3 (range 9)))
         tens-b (ct/->tensor (partition 3 (repeat 9 1)))]

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
         tens-b (ct/->tensor (partition 3 (repeat 9 2)))
         tens-c (ct/->tensor (partition 3 (repeat 9 10)))]
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
                       (ct/to-double-array tens-c-small)))))
     (when (contains? #{:float :double} datatype)
       (let [n-batches 3
             n-channels 5
             img-dim 4
             big-pools (repeatedly (*  n-channels n-batches)
                                   (fn []
                                     (vec (repeatedly (* img-dim img-dim) rand))))
             sums (->> (mapv #(apply + %) big-pools)
                       (partition n-channels)
                       (apply m/add))

             big-m (ct/->tensor (->> (flatten big-pools)
                                     (partition img-dim)
                                     (partition img-dim)
                                     (partition n-channels)))
             test-vec (-> (ct/new-tensor [n-channels])
                          (ct/in-place-reshape [n-channels 1 1]))]
         ;;broadcasting summation
         (ct/binary-op! test-vec 1.0 test-vec 1.0 big-m :+)
         (is (m/equals sums
                       (ct/to-double-array test-vec)
                       1e-4))))

     (let [tens-a (ct/->tensor (repeat 4 (partition 3 (range 9))))
           result (ct/new-tensor (m/shape tens-a))]
       (ct/binary-op! result 1.0 tens-a 1.0 5 :eq)
       (is (m/equals (mapv #(if (= (long %) 5)
                              1
                              0)
                           (ct/to-double-array tens-a))
                     (ct/to-double-array result))))

     (let [tens-a (ct/new-tensor [4 3 3])
           bias (-> (ct/->tensor [1 2 3 4])
                    (ct/in-place-reshape [4 1 1]))]
       (ct/binary-op! tens-a 1.0 tens-a 1.0 bias :+)
       (is (m/equals (flatten
                      (map #(repeat 9 %) [1 2 3 4]))
                     (ct/to-double-array tens-a))))
     ;;bias-gradient calculation
     (let [tens-a (ct/->tensor  (mapv #(->> (repeat 9 %)
                                            (partition 3)) [1 2 3 4]))
           bias (-> (ct/new-tensor [4])
                    (ct/in-place-reshape [4 1 1]))]
       (ct/binary-op! bias 1.0 tens-a 1.0 bias :+)
       (is (m/equals [9 18 27 36]
                     (ct/to-double-array bias))))

     (let [tens-1 (ct/->tensor [1 1 0 0])
           tens-2 (ct/->tensor [-1 1 2 -2])
           result (ct/new-tensor (m/shape tens-2))]

        ;;; > greater than
       (ct/binary-op! result 1.0 tens-1 1.0 tens-2 :>)
       (is (m/equals [1 0 0 1]
                     (ct/to-double-array result)))

       ;;; >= greater than or equal to
       (ct/binary-op! result 1.0 tens-1 1.0 tens-2 :>=)
       (is (m/equals [1 1 0 1]
                     (ct/to-double-array result)))

       ;;; < less than
       (ct/binary-op! result 1.0 tens-1 1.0 tens-2 :<)
       (is (m/equals [0 0 1 0]
                     (ct/to-double-array result)))

       ;;; <= less than or equal to
       (ct/binary-op! result 1.0 tens-1 1.0 tens-2 :<=)
       (is (m/equals [0 1 1 0]
                     (ct/to-double-array result)))))

   ;; bit-xor
   (let [tens-1 (ct/->tensor [1 1 0 0])
         tens-2 (ct/->tensor [1 1 1 1])
         result (ct/new-tensor (m/shape tens-2))]

       (ct/binary-op! result 1.0 tens-1 1.0 tens-2 :bit-xor)
       (is (m/equals [0 0 1 1]
                     (ct/to-double-array result))))))


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


(defn gemv
  [driver datatype]
  (tensor-context
   driver datatype
   (let [tens-a (ct/->tensor (partition 3 (range 12)))
         tens-b (ct/->tensor (repeat 4 2))
         tens-c (ct/->tensor (range 4))
         tens-b-sub (ct/subvector tens-b 0 :length 3)]
     (ct/gemv! tens-c false 1 tens-a tens-b 1)
     (is (m/equals [6.0 25.0 44.0 63.0]
                   (ct/to-double-array tens-c)))
     (ct/gemv! tens-c false 1 tens-a tens-b 0)
     (is (m/equals [6 24 42 60]
                   (ct/to-double-array tens-c)))
     (let [tens-c-sub (ct/subvector tens-c 0 :length 3)
           ;;[1 4 7 10]
           tens-a-col (second (ct/columns tens-a))]
       (ct/gemv! tens-c-sub true 1 tens-a tens-a-col 0)
       (is (m/equals [144.0 166.0 188.0]
                     (ct/to-double-array tens-c-sub)))))))


(defn batch-normalize
  [driver datatype]
  (tensor-context
   driver datatype
   ;;eltwise
   (let [input (ct/->tensor (partition 3 (range 12)))
         output (ct/new-tensor [4 3])
         means (ct/->tensor (repeat 3 1))
         ;;Ensure to include a variance of 0 to sniff out behavior in edge case
         variances (ct/->tensor (range 0 3))
         scale (ct/->tensor (repeat 3 3))
         bias (ct/->tensor (repeat 3 4))]
     ;;Use a large epsilon to ensure the cpu version treats the epsilon identical
     ;;as the gpu version
     (ct/batch-normalize! output input means variances scale bias 1e-2)
     (is (m/equals [-26.0, 4.0, 6.116036847575795, 64.0, 12.955334711889902,
                    12.46414739030318, 154.0, 21.910669423779805, 18.812257933030565,
                    244.0, 30.86600413566971, 25.16036847575795]
                   (ct/to-double-array output)
                   1e-4)))
   ;;spatial
   (let [input (ct/->tensor (partition 3 (partition 4 (range 24))))
         output (ct/new-tensor [2 3 4])
         means (ct/->tensor (repeat 3 1))
         ;;Ensure to include a variance of 0
         variances (ct/->tensor (range 0 3))
         scale (ct/->tensor (repeat 3 3))
         bias (ct/->tensor (repeat 3 4))]
     ;;Use a large epsilon to ensure the cpu version treats the epsilon identical
     ;;as the gpu version
     (ct/batch-normalize! output input means variances scale bias 1e-2)
     (is (m/equals [-26.0, 4.0, 34.0, 64.0, 12.955334711889902, 15.94044628251987,
                    18.925557853149837, 21.910669423779805, 18.812257933030565,
                    20.92829478060636, 23.044331628182157, 25.16036847575795, 334.0,
                    364.0, 394.0, 424.0, 48.77667355944951, 51.76178513007948,
                    54.74689670070945, 57.73200827133942, 44.204700103940105,
                    46.3207369515159, 48.436773799091696, 50.55281064666749]
                   (ct/to-double-array output)
                   1e-4)))))

(defn batch-normalize-update-and-apply
  [driver datatype]
  (tensor-context
   driver datatype
   ;;eltwise
   (let [input (ct/->tensor (partition 3 (range 12)))
         output (ct/new-tensor [4 3])
         means (ct/->tensor (repeat 3 1))
         ;;Ensure to include a variance of 0 to sniff out behavior in edge case
         variances (ct/->tensor (range 0 3))
         running-means (ct/->tensor (repeat 3 2))
         running-variances (ct/->tensor (repeat 3 4))
         ave-factor 0.8
         scale (ct/->tensor (repeat 3 3))
         bias (ct/->tensor (repeat 3 4))]
     ;;Use a large epsilon to ensure the cpu version treats the epsilon identical
     ;;as the gpu version
     (ct/batch-normalize-update-and-apply! output input means variances
                                           running-means running-variances ave-factor
                                           scale bias 1e-2)
     (is (m/equals [-0.023134696804511745, -0.023134696804511745, -0.023134696804511745,
                    2.6589551010651626, 2.6589551010651626, 2.6589551010651626,
                    5.341044898934837, 5.341044898934837, 5.341044898934837,
                    8.023134696804512, 8.023134696804512, 8.023134696804512]
                   (ct/to-double-array output)
                   1e-4))
     (is (m/equals [4.5, 5.5, 6.5]
                   (ct/to-double-array means)
                   1e-4))
     ;;NVIDIA stores the batch variances in an odd 1/sqrt form.  This allows them to
     ;;compute the answer slightly faster but it means the actual meaning of the
     ;;variances variable is obscured.  Thus we cannot reliably test the batch-variances
     ;;variable across implementations.  If you want per-batch variances then you need
     ;;to set the average factor to 1.0.
     (comment
       (is (m/equals [0.29800997754107494, 0.29800997754107494, 0.29800997754107494]
                     (ct/to-double-array variances)
                     1e-4)))
     (is (m/equals [4.0, 4.8, 5.6000000000000005]
                   (ct/to-double-array running-means)
                   1e-4))
     (is (m/equals [12.8, 12.8, 12.8]
                   (ct/to-double-array running-variances)
                   1e-4)))
   ;;spatial
   (let [input (ct/->tensor (partition 3 (partition 4 (range 24))))
         output (ct/new-tensor [2 3 4])
         means (ct/->tensor (repeat 3 1))
         ;;Ensure to include a variance of 0
         variances (ct/->tensor (range 0 3))
         running-means (ct/->tensor (repeat 3 2))
         running-variances (ct/->tensor (repeat 3 4))
         ave-factor 0.8
         scale (ct/->tensor (repeat 3 3))
         bias (ct/->tensor (repeat 3 4))]
     ;;Use a large epsilon to ensure the cpu version treats the epsilon identical
     ;;as the gpu version
     (ct/batch-normalize-update-and-apply! output input means variances
                                           running-means running-variances
                                           ave-factor scale bias 1e-2)
     (is (m/equals
          [0.3139510961275713, 0.8054242833105618, 1.2968974704935523,
           1.7883706576765428, 0.3139510961275713, 0.8054242833105618,
           1.2968974704935523, 1.7883706576765428, 0.3139510961275713,
           0.8054242833105618, 1.2968974704935523, 1.7883706576765428,
           6.211629342323457, 6.703102529506448, 7.194575716689438,
           7.686048903872429, 6.211629342323457, 6.703102529506448,
           7.194575716689438, 7.686048903872429, 6.211629342323457,
           6.703102529506448, 7.194575716689438, 7.686048903872429]
          (ct/to-double-array output)
          1e-4))
     (is (m/equals [7.5, 11.5, 15.5]
                   (ct/to-double-array means)
                   1e-4))
     (is (m/equals [6.4, 9.6, 12.8]
                   (ct/to-double-array running-means)
                   1e-4))
     (is (m/equals [34.857142857142854, 34.857142857142854, 34.857142857142854]
                   (ct/to-double-array running-variances)
                   1e-4)))))


(defn batch-normalize-gradients
  [driver datatype]
  (tensor-context
   driver datatype
   ;;eltwise
   (let [input (ct/->tensor (partition 3 (range 12)))
         output (ct/->tensor (partition
                              3 [-26.0, 4.0, 6.116036847575795, 64.0, 12.955334711889902
                                 12.46414739030318, 154.0, 21.910669423779805, 18.812257933030565
                                 244.0, 30.86600413566971, 25.16036847575795]))
         input-gradient (ct/new-tensor [4 3])
         output-gradient (ct/->tensor (partition 3 (range 12)))
         batch-means (ct/->tensor (repeat 3 1))
         ;;Ensure to include a variance of 0 to sniff out behavior in edge case
         batch-variances (ct/->tensor (range 0 3))
         running-means (ct/->tensor (repeat 3 2))
         running-variances (ct/->tensor (repeat 3 4))
         scale (ct/->tensor (repeat 3 3))
         bias (ct/->tensor (repeat 3 4))
         scale-gradient (ct/new-tensor [3])
         bias-gradient (ct/new-tensor [3])
         ave-factor 0.8]
     ;;Use a large epsilon to ensure the cpu version treats the epsilon identical
     ;;as the gpu version
     (ct/batch-normalize-update-and-apply! output input batch-means batch-variances
                                           running-means running-variances ave-factor
                                           scale bias 1e-2)
     (ct/batch-normalize-gradients! input-gradient scale-gradient bias-gradient output-gradient
                                    output input batch-means batch-variances scale bias 1e-2)
     (is (m/equals [-0.023134696804511745, -0.023134696804511745, -0.023134696804511745,
                    2.6589551010651626, 2.6589551010651626, 2.6589551010651626,
                    5.341044898934837, 5.341044898934837, 5.341044898934837,
                    8.023134696804512, 8.023134696804512, 8.023134696804512]
                   (ct/to-double-array output)
                   1e-4))
     (is (m/equals [-0.003572943780465465, -0.003572943780465465, -0.003572943780465465,
                    -0.001190981260155155, -0.001190981260154711, -0.001190981260154489,
                    0.001190981260154933, 0.001190981260155155, 0.0011909812601555991,
                    0.0035729437804663533, 0.0035729437804663533, 0.0035729437804663533]
                   (ct/to-double-array input-gradient)
                   1e-4))
     (is (m/equals [13.410448989348373, 13.410448989348373, 13.410448989348373]
                   (ct/to-double-array scale-gradient)
                   1e-4))
     (is (m/equals [18 22 26]
                   (ct/to-double-array bias-gradient)
                   1e-4)))
   ;;spatial
   (let [input (ct/->tensor (partition 3 (partition 4 (range 24))))
         output (ct/new-tensor [2 3 4])
         input-gradient (ct/new-tensor [2 3 4])
         output-gradient (ct/->tensor (->> (range 24)
                                           (partition 4)
                                           (partition 3)))
         batch-means (ct/->tensor (repeat 3 1))
         ;;Ensure to include a variance of 0
         batch-variances (ct/->tensor (range 0 3))
         running-means (ct/->tensor (repeat 3 2))
         running-variances (ct/->tensor (repeat 3 4))
         ave-factor 0.8
         scale (ct/->tensor (repeat 3 3))
         bias (ct/->tensor (repeat 3 4))
         scale-gradient (ct/new-tensor [3])
         bias-gradient (ct/new-tensor [3])]
     ;;Use a large epsilon to ensure the cpu version treats the epsilon identical
     ;;as the gpu version
     (ct/batch-normalize-update-and-apply! output input batch-means batch-variances
                                           running-means running-variances
                                           ave-factor scale bias 1e-2)
     (ct/batch-normalize-gradients! input-gradient scale-gradient bias-gradient output-gradient
                                    output input batch-means batch-variances scale bias 1e-2)
     (is (m/equals
          [0.3139510961275713, 0.8054242833105618, 1.2968974704935523,
           1.7883706576765428, 0.3139510961275713, 0.8054242833105618,
           1.2968974704935523, 1.7883706576765428, 0.3139510961275713,
           0.8054242833105618, 1.2968974704935523, 1.7883706576765428,
           6.211629342323457, 6.703102529506448, 7.194575716689438,
           7.686048903872429, 6.211629342323457, 6.703102529506448,
           7.194575716689438, 7.686048903872429, 6.211629342323457,
           6.703102529506448, 7.194575716689438, 7.686048903872429]
          (ct/to-double-array output)
          1e-4))
     (is (m/equals
          [-9.892777519776665E-4, -8.57374051714065E-4, -7.254703514504634E-4,
           -5.935666511868618E-4, -9.89277751978103E-4, -8.57374051714065E-4,
           -7.254703514500268E-4, -5.935666511868618E-4, -9.89277751978103E-4,
           -8.57374051714065E-4, -7.254703514500268E-4, -5.935666511868618E-4,
           5.935666511868618E-4, 7.254703514504634E-4, 8.57374051714065E-4,
           9.892777519776665E-4, 5.935666511868618E-4, 7.254703514500268E-4,
           8.57374051714065E-4, 9.89277751978103E-4, 5.935666511868618E-4,
           7.254703514500268E-4, 8.57374051714065E-4, 9.89277751978103E-4]
          (ct/to-double-array input-gradient)
          1e-6))
     (is (m/equals [48.81966992684372, 48.81966992684372, 48.81966992684372]
                   (ct/to-double-array scale-gradient)
                   1e-4))
     (is (m/equals [60.0, 92.0, 124.0]
                   (ct/to-double-array bias-gradient)
                   1e-4)))))


(defn activation-forward
  [driver datatype]
  (tensor-context
   driver datatype
   (let [input (ct/->tensor (range -4 8))
         output (ct/new-tensor [12])]
     (ct/unary-op! output 1.0 input :logistic)
     (is (m/equals [0.01798620996209156, 0.04742587317756678, 0.11920292202211755,
                    0.2689414213699951, 0.5, 0.7310585786300049, 0.8807970779778823,
                    0.9525741268224334, 0.9820137900379085, 0.9933071490757153,
                    0.9975273768433653, 0.9990889488055994]
                   (ct/to-double-array output)
                   1e-4))
     (ct/unary-op! output 1.0 input :tanh)
     (is (m/equals [-0.999329299739067, -0.9950547536867305, -0.9640275800758169,
                    -0.7615941559557649, 0.0, 0.7615941559557649, 0.9640275800758169,
                    0.9950547536867305, 0.999329299739067, 0.9999092042625951,
                    0.9999877116507956, 0.9999983369439447]
                   (ct/to-double-array output)
                   1e-4))
     (ct/binary-op! output 1.0 input 0 0 :max)
     (is (m/equals [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
                   (ct/to-double-array output))))))


(defn activation-gradient
  [driver datatype]
  (tensor-context
   driver datatype
   (let [output (ct/->tensor (range 12))
         output-gradient (ct/->tensor (range -4 8))
         input-gradient (ct/new-tensor [12])]
     (ct/activation-gradient! input-gradient output-gradient output :logistic)
     (is (m/equals [-0.0, -0.0, 4.0, 6.0, -0.0, -20.0, -60.0,
                    -126.0, -224.0, -360.0, -540.0, -770.0]
                   (ct/to-double-array input-gradient)))
     (ct/activation-gradient! input-gradient output-gradient output :tanh)
     (is (m/equals [-4.0, -0.0, 6.0, 8.0, -0.0, -24.0, -70.0,
                    -144.0, -252.0, -400.0, -594.0, -840.0]
                   (ct/to-double-array input-gradient)))
     (ct/activation-gradient! input-gradient output-gradient output :relu)
     (is (m/equals [0.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
                   (ct/to-double-array input-gradient))))))


(defn softmax
  [driver datatype]
  (tensor-context
   driver datatype
   (let [input (ct/->tensor (repeat 10 [1 2 3 4]))
         output (ct/new-tensor [10 4])]
     (ct/softmax! output input)
     (is (m/equals   [0.03205860328008499, 0.08714431874203256, 0.2368828180899101,
                      0.6439142598879722, 0.03205860328008499, 0.08714431874203256,
                      0.2368828180899101, 0.6439142598879722, 0.03205860328008499,
                      0.08714431874203256, 0.2368828180899101, 0.6439142598879722,
                      0.03205860328008499, 0.08714431874203256, 0.2368828180899101,
                      0.6439142598879722, 0.03205860328008499, 0.08714431874203256,
                      0.2368828180899101, 0.6439142598879722, 0.03205860328008499,
                      0.08714431874203256, 0.2368828180899101, 0.6439142598879722,
                      0.03205860328008499, 0.08714431874203256, 0.2368828180899101,
                      0.6439142598879722, 0.03205860328008499, 0.08714431874203256,
                      0.2368828180899101, 0.6439142598879722, 0.03205860328008499,
                      0.08714431874203256, 0.2368828180899101, 0.6439142598879722,
                      0.03205860328008499, 0.08714431874203256, 0.2368828180899101,
                      0.6439142598879722]
                     (ct/to-double-array output)
                     1e-4)))
   (let [img-dim 4
         input (ct/->tensor [(->> [1 2 3 4]
                                   (mapv #(repeat (* img-dim img-dim) %)))])
         output (ct/new-tensor [1 4 (* img-dim img-dim)])]
     (ct/softmax! output input)
     (is (m/equals   (->> [0.03205860328008499 0.08714431874203257
                           0.23688281808991013 0.6439142598879724]
                          (mapv #(repeat (* img-dim img-dim) %))
                          flatten
                          vec)
                     (ct/to-double-array output)
                     1e-4)))
   ;;Some loss terms want to use softmax with strides that aren't contiguous
   (let [pred-matrix (ct/->tensor (repeat 5 (range 10)))
         sel-pred-matr (ct/select pred-matrix :all (range 2 10))
         test-tensor (ct/->tensor (range 2 10))
         answer (ct/softmax! test-tensor test-tensor)]
     (ct/softmax! sel-pred-matr sel-pred-matr)
     (is (m/equals (flatten (repeat 5 (concat [0 1]
                                              (ct/to-double-array answer))))
                   (ct/to-double-array pred-matrix))))))


(defn ternary-op-select
  [driver datatype]
  (tensor-context
   driver datatype
   (let [dest (ct/->tensor (repeat 10 0))
         x-arg (ct/->tensor (range -5 5))
         y-arg (ct/->tensor (range 10))
         z-arg (ct/->tensor (repeat 10 2))]
     (ct/ternary-op! dest 1 x-arg 2.0 y-arg 3.0 z-arg :select)
     (is (m/equals [0 2 4 6 8 6 6 6 6 6]
                   (ct/to-double-array dest)))
     (ct/ternary-op! dest 1 x-arg 1.0 -1 3.0 z-arg :select)
     (is (m/equals [-1 -1 -1 -1 -1 6 6 6 6 6]
                   (ct/to-double-array dest)))
     (ct/ternary-op! dest 1 x-arg 3.0 z-arg 1.0 -1 :select)
     (is (m/equals [6 6 6 6 6 -1 -1 -1 -1 -1]
                   (ct/to-double-array dest)))
     (ct/ternary-op! dest 1 x-arg 3.0 2.0 1.0 -1 :select)
     (is (m/equals [6 6 6 6 6 -1 -1 -1 -1 -1]
                   (ct/to-double-array dest)))
     (ct/ternary-op! dest 1 x-arg 1.0 -1 3.0 2.0 :select)
     (is (m/equals [-1 -1 -1 -1 -1 6 6 6 6 6]
                   (ct/to-double-array dest))))))


(defn unary-reduce
  [driver datatype]
  (tensor-context
   driver datatype
   (let [dest (ct/new-tensor [10 1])
         src-data [0 3 5 2 1 9 5 7 7 2]
         src (ct/->tensor (repeat 10 src-data))]
     (ct/unary-reduce! dest 2.0 src :max)
     (is (m/equals (repeat 10 18)
                   (ct/to-double-array dest)))
     (ct/unary-reduce! dest 1.0 src :sum)
     (is (m/equals (repeat 10 (apply + src-data))
                   (ct/to-double-array dest)))
     (ct/unary-reduce! dest 1.0 src :mean)
     (is (m/equals (repeat 10 (drv/dtype-cast (/ (apply + src-data)
                                                 (count src-data))
                                              datatype))
                   (ct/to-double-array dest))))))


(defn transpose
  [driver datatype]
  (tensor-context
   driver datatype
   (let [img-dim 4
         img-tensor (ct/->tensor
                     (->> (repeat (* img-dim img-dim) [1 2 3])
                          (partition img-dim)))
         planar-tensor (ct/transpose img-tensor [2 0 1])
         rgb-tensor (ct/transpose planar-tensor [1 2 0])]
     (is (m/equals (flatten (concat (repeat (* img-dim img-dim) 1)
                                    (repeat (* img-dim img-dim) 2)
                                    (repeat (* img-dim img-dim) 3)))
                   (ct/to-double-array planar-tensor)))

     (is (m/equals (flatten (repeat (* img-dim img-dim) [1 2 3]))
                   (ct/to-double-array rgb-tensor))))))


(defn mask
  [driver datatype]
  (tensor-context
   driver datatype
   (let [r-pix (int 1)
         g-pix (int 2)
         b-pix (int 3)
         ;;Load a single image to r,g,b planes
         rgba (+ r-pix
                 (bit-shift-left g-pix 8)
                 (bit-shift-left b-pix 16)
                 (bit-shift-left (int 255) 24))
         img-dim 4
         img-tensor (ct/->tensor
                     (->> (repeat (* img-dim img-dim) rgba)
                          (partition img-dim)))
         mask-tensor (-> (ct/->tensor [0xFF
                                       (bit-shift-left 0xFF 8)
                                       (bit-shift-left 0xFF 16)])
                         (ct/in-place-reshape [3 1 1]))
         div-tensor (-> (ct/->tensor [1
                                      (bit-shift-left 1 8)
                                      (bit-shift-left 1 16)])
                        (ct/in-place-reshape [3 1 1]))
         result (ct/new-tensor [3 img-dim img-dim])]
     (ct/binary-op! result 1.0 img-tensor 1.0 mask-tensor :bit-and)
     (ct/binary-op! result 1.0 result 1.0 div-tensor :/)
     (is (m/equals (flatten (concat (repeat (* img-dim img-dim) 1)
                                    (repeat (* img-dim img-dim) 2)
                                    (repeat (* img-dim img-dim) 3)))
                   (ct/to-double-array result))))))


(defn select
  [driver datatype]
  (tensor-context
   driver datatype
   (let [mat-tens (ct/->tensor (repeat 2 (partition 3 (range 9))))]
     (let [sel-tens (ct/select mat-tens :all :all [1 2])]
       (is (m/equals (flatten (repeat 2 [1 2 4 5 7 8]))
                     (ct/to-double-array sel-tens)))
       (is (m/equals [2 3 2]
                     (m/shape sel-tens))))
     (let [sel-tens (ct/select mat-tens :all :all [2])]
       (is (m/equals (flatten (repeat 2 [2 5 8]))
                     (ct/to-double-array sel-tens)))
       (is (m/equals [2 3 1]
                     (m/shape sel-tens))))
     (let [sel-tens (ct/select mat-tens :all :all 2)]
       (is (m/equals (flatten (repeat 2 [2 5 8]))
                     (ct/to-double-array sel-tens)))
       (is (m/equals [2 3]
                     (m/shape sel-tens)))
       (is (not (ct/dense? sel-tens))))

     (let [sel-tens (ct/select mat-tens :all [1 2] :all)]
       (is (m/equals (flatten (repeat 2 [3 4 5 6 7 8]))
                     (ct/to-double-array sel-tens)))
       (is (m/equals [2 2 3]
                     (m/shape sel-tens)))
       (is (not (ct/dense? sel-tens))))
     (let [sel-tens (ct/select mat-tens :all [2] :all)]
       (is (m/equals (flatten (repeat 2 [6 7 8]))
                     (ct/to-double-array sel-tens)))
       (is (m/equals [2 1 3]
                     (m/shape sel-tens)))
       (is (not (ct/dense? sel-tens))))

     (let [sel-tens (ct/select mat-tens :all 0 :all)]
       (is (m/equals (flatten (repeat 2 [0 1 2]))
                     (ct/to-double-array sel-tens)))
       (is (m/equals [2 3]
                     (m/shape sel-tens)))
       (is (not (ct/dense? sel-tens))))

     (let [sel-tens (ct/select mat-tens [1] [1] :all)]
       (is (m/equals [3 4 5]
                     (ct/to-double-array sel-tens)))
       (is (m/equals [1 1 3]
                     (m/shape sel-tens)))
       (is (ct/dense? sel-tens)))

     (let [sel-tens (ct/select mat-tens 1 1 :all)]
       (is (m/equals [3 4 5]
                     (ct/to-double-array sel-tens)))
       (is (m/equals [3]
                     (m/shape sel-tens)))
       (is (ct/dense? sel-tens))
       (is (ct/as-vector sel-tens)))

     (let [sel-tens (ct/select mat-tens 1 :all 2)]
       (is (m/equals [2 5 8]
                     (ct/to-double-array sel-tens)))
       (is (m/equals [3]
                     (m/shape sel-tens)))
       (is (not (ct/dense? sel-tens)))))))


(defn select-transpose-interaction
  [driver datatype]
  (tensor-context
   driver datatype
   (let [img-dim 4
         mat-tens (ct/->tensor (partition img-dim (repeat (* img-dim img-dim) [1 2 3])))
         planar-tens (ct/transpose mat-tens [2 0 1])
         n-pixels (* img-dim img-dim)]
     (let [r-tens (ct/select planar-tens 0 :all :all)
           g-tens (ct/select planar-tens 1 :all :all)
           b-tens (ct/select planar-tens 2 :all :all)]
       (is (m/equals (repeat n-pixels 1) (ct/to-double-array r-tens)))
       (is (m/equals (repeat n-pixels 2) (ct/to-double-array g-tens)))
       (is (m/equals (repeat n-pixels 3) (ct/to-double-array b-tens)))
       (let [bgr-tens (ct/new-tensor [img-dim img-dim 3])
             bgr-planes (ct/transpose bgr-tens [2 0 1])]
         (m/assign! (ct/select bgr-planes 0 :all :all) b-tens)
         (m/assign! (ct/select bgr-planes 1 :all :all) g-tens)
         (m/assign! (ct/select bgr-planes 2 :all :all) r-tens)
         (is (m/equals (flatten (partition img-dim (repeat (* img-dim img-dim) [3 2 1])))
                       (ct/to-double-array bgr-tens))))))))


(defn convolution-operator
  [driver datatype]
  (tensor-context
   driver datatype
   (let [batch-size 3
         num-in-channels 4
         num-out-channels 3
         input-dim 4
         input (ct/->tensor (->> (range (* input-dim input-dim batch-size num-in-channels))
                                 (partition input-dim)
                                 (partition input-dim)
                                 (partition num-in-channels)))
         conv-desc (ct/convolution-descriptor datatype num-out-channels num-in-channels
                                              3 3 0 0 1 1)
         {:keys [output-width output-height]} (ct/get-convolution-output-dimensions conv-desc input-dim input-dim)
         output-width (long output-width)
         output-height (long output-height)
         output (ct/new-tensor [batch-size num-out-channels output-height output-width])
         output-gradient (-> (ct/->tensor (repeat (* batch-size
                                                     output-width output-height
                                                     num-out-channels) 1))
                             (ct/in-place-reshape [batch-size num-out-channels
                                                   output-height output-width]))
         algorithms (ct/choose-convolution-algorithms conv-desc input-dim input-dim batch-size 1000)
         workspace (ct/new-tensor [(long (get algorithms :workspace-size))])
         weights (ct/->tensor (take num-out-channels
                                    (partition (* 3 3 num-in-channels) (range))))
         bias-gradient (ct/new-tensor [num-out-channels])
         weight-gradient (ct/new-tensor (ct/shape weights))
         input-gradient (ct/new-tensor (ct/shape input))]
     ;;Make sure that we test out pre-existing conditions.
     (m/assign! output 1)
     (m/assign! weight-gradient 1)
     (m/assign! input-gradient 1)
     (ct/convolution-forward! output 0.0 input weights workspace conv-desc algorithms)
     (is (m/equals   [25062.0, 25692.0, 27582.0, 28212.0, 62646.0, 64572.0, 70350.0,
                      72276.0, 100230.0, 103452.0, 113118.0, 116340.0, 65382.0, 66012.0,
                      67902.0, 68532.0, 185910.0, 187836.0, 193614.0, 195540.0, 306438.0,
                      309660.0, 319326.0, 322548.0, 105702.0, 106332.0, 108222.0,
                      108852.0, 309174.0, 311100.0, 316878.0, 318804.0, 512646.0,
                      515868.0, 525534.0, 528756.0]
                     (ct/to-double-array output)))

     (ct/convolution-backward-weights! weight-gradient 0.0 output-gradient input workspace conv-desc algorithms)
     (is (m/equals   [798.0, 810.0, 822.0, 846.0, 858.0, 870.0, 894.0, 906.0, 918.0,
                      990.0, 1002.0, 1014.0, 1038.0, 1050.0, 1062.0, 1086.0, 1098.0,
                      1110.0, 1182.0, 1194.0, 1206.0, 1230.0, 1242.0, 1254.0, 1278.0,
                      1290.0, 1302.0, 1374.0, 1386.0, 1398.0, 1422.0, 1434.0, 1446.0,
                      1470.0, 1482.0, 1494.0, 798.0, 810.0, 822.0, 846.0, 858.0, 870.0,
                      894.0, 906.0, 918.0, 990.0, 1002.0, 1014.0, 1038.0, 1050.0, 1062.0,
                      1086.0, 1098.0, 1110.0, 1182.0, 1194.0, 1206.0, 1230.0, 1242.0,
                      1254.0, 1278.0, 1290.0, 1302.0, 1374.0, 1386.0, 1398.0, 1422.0,
                      1434.0, 1446.0, 1470.0, 1482.0, 1494.0, 798.0, 810.0, 822.0, 846.0,
                      858.0, 870.0, 894.0, 906.0, 918.0, 990.0, 1002.0, 1014.0, 1038.0,
                      1050.0, 1062.0, 1086.0, 1098.0, 1110.0, 1182.0, 1194.0, 1206.0,
                      1230.0, 1242.0, 1254.0, 1278.0, 1290.0, 1302.0, 1374.0, 1386.0,
                      1398.0, 1422.0, 1434.0, 1446.0, 1470.0, 1482.0, 1494.0]
                   (ct/to-double-array weight-gradient)))

     (ct/convolution-backward-data! input-gradient 0.0 output-gradient weights workspace conv-desc algorithms)
     (is (m/equals [108.0, 219.0, 225.0, 114.0, 225.0, 456.0, 468.0, 237.0, 243.0,
                    492.0, 504.0, 255.0, 126.0, 255.0, 261.0, 132.0, 135.0, 273.0,
                    279.0, 141.0, 279.0, 564.0, 576.0, 291.0, 297.0, 600.0, 612.0,
                    309.0, 153.0, 309.0, 315.0, 159.0, 162.0, 327.0, 333.0, 168.0,
                    333.0, 672.0, 684.0, 345.0, 351.0, 708.0, 720.0, 363.0, 180.0,
                    363.0, 369.0, 186.0, 189.0, 381.0, 387.0, 195.0, 387.0, 780.0,
                    792.0, 399.0, 405.0, 816.0, 828.0, 417.0, 207.0, 417.0, 423.0,
                    213.0, 108.0, 219.0, 225.0, 114.0, 225.0, 456.0, 468.0, 237.0,
                    243.0, 492.0, 504.0, 255.0, 126.0, 255.0, 261.0, 132.0, 135.0,
                    273.0, 279.0, 141.0, 279.0, 564.0, 576.0, 291.0, 297.0, 600.0,
                    612.0, 309.0, 153.0, 309.0, 315.0, 159.0, 162.0, 327.0, 333.0,
                    168.0, 333.0, 672.0, 684.0, 345.0, 351.0, 708.0, 720.0, 363.0,
                    180.0, 363.0, 369.0, 186.0, 189.0, 381.0, 387.0, 195.0, 387.0,
                    780.0, 792.0, 399.0, 405.0, 816.0, 828.0, 417.0, 207.0, 417.0,
                    423.0, 213.0, 108.0, 219.0, 225.0, 114.0, 225.0, 456.0, 468.0,
                    237.0, 243.0, 492.0, 504.0, 255.0, 126.0, 255.0, 261.0, 132.0,
                    135.0, 273.0, 279.0, 141.0, 279.0, 564.0, 576.0, 291.0, 297.0,
                    600.0, 612.0, 309.0, 153.0, 309.0, 315.0, 159.0, 162.0, 327.0,
                    333.0, 168.0, 333.0, 672.0, 684.0, 345.0, 351.0, 708.0, 720.0,
                    363.0, 180.0, 363.0, 369.0, 186.0, 189.0, 381.0, 387.0, 195.0,
                    387.0, 780.0, 792.0, 399.0, 405.0, 816.0, 828.0, 417.0, 207.0,
                    417.0, 423.0, 213.0]
                   (ct/to-double-array input-gradient))))))


(defn pooling-operator
  [driver datatype]
  (tensor-context
    driver datatype
    (let [channels 4
          dimension 4
          batch-size 2
          n-elems (* batch-size channels dimension dimension)
          input (ct/->tensor (->> (range -5 15)
                                  (repeat)
                                  (apply concat)
                                  (take n-elems)
                                  (partition dimension)
                                  (partition dimension)
                                  (partition channels)))
          max-pooling-desc (tensor/pooling-descriptor datatype channels 3 3 1 1 2 2
                                                      :pool-op :max
                                                      :dimension-op :floor)
          avg-pooling-desc (tensor/pooling-descriptor datatype channels 3 3 1 1 2 2
                                                      :pool-op :avg
                                                      :dimension-op :floor)
          avg-exc-pad-pooling-desc (tensor/pooling-descriptor datatype channels 3 3 1 1 2 2
                                                              :pool-op :avg-exc-pad
                                                              :dimension-op :floor)
          {:keys [output-width output-height]} (tensor/get-convolution-output-dimensions
                                                 max-pooling-desc
                                                 dimension dimension)
          max-output (ct/new-tensor [batch-size channels output-width output-height])
          avg-output (ct/new-tensor [batch-size channels output-width output-height])
          avg-exc-pad-output (ct/new-tensor [batch-size channels output-width output-height])
          max-input-gradient (ct/new-tensor (m/shape input))
          avg-input-gradient (ct/new-tensor (m/shape input))
          avg-exc-pad-input-gradient (ct/new-tensor (m/shape input))
          output-gradient (->> (repeat (* batch-size channels output-width output-height) 1)
                               (partition output-width)
                               (partition output-height)
                               (partition channels)
                               (ct/->tensor))]
      (ct/pooling-forward! max-output 0.0 input max-pooling-desc)
      (ct/pooling-forward! avg-output 0.0 input avg-pooling-desc)
      (ct/pooling-forward! avg-exc-pad-output 0.0 input avg-exc-pad-pooling-desc)
      (testing "max pool forward"
        (is (m/equals [0.0 2.0 8.0 10.0 12.0 14.0 4.0 6.0 12.0
                       14.0 12.0 14.0 8.0 10.0 12.0 14.0 4.0 6.0
                       12.0 14.0 0.0 2.0 8.0 10.0 12.0 14.0 4.0
                       6.0 12.0 14.0 12.0 14.0]
                      (vec (ct/to-double-array max-output)))))
      (testing "avg pool forward"
        (is (m/equals [-1.1111111111111112 -0.6666666666666666 2.3333333333333335 5.0
                       1.5555555555555556 3.3333333333333335 -0.3333333333333333 1.0
                       4.222222222222222 7.333333333333333 1.4444444444444444
                       3.6666666666666665 2.4444444444444446 4.666666666666667
                       3.2222222222222223 6.333333333333333 0.6666666666666666 2.0 5.0
                       9.0 -1.1111111111111112 -0.6666666666666666 2.3333333333333335 5.0
                       1.5555555555555556 3.3333333333333335 -0.3333333333333333 1.0
                       4.222222222222222 7.333333333333333 1.4444444444444444 3.6666666666666665]
                      (vec (ct/to-double-array avg-output))
                      1e-4)))
      (testing "avg-exc-pad pool forward"
        (is (m/equals [-2.5 -1.0 3.5 5.0 3.5 5.0 -0.5 1.0 9.5 11.0
                       2.1666666666666665 3.6666666666666665 5.5 7.0 4.833333333333333
                       6.333333333333333 1.5 3.0 7.5 9.0 -2.5 -1.0 3.5 5.0 3.5 5.0
                       -0.5 1.0 9.5 11.0 2.1666666666666665 3.6666666666666665]
                      (vec (ct/to-double-array avg-exc-pad-output))
                      1e-4)))
      (ct/pooling-backward! max-input-gradient 0.0 input max-output output-gradient max-pooling-desc)
      (ct/pooling-backward! avg-input-gradient 0.0 input avg-output output-gradient avg-pooling-desc)
      (ct/pooling-backward! avg-exc-pad-input-gradient 0.0 input avg-exc-pad-output
                            output-gradient avg-exc-pad-pooling-desc)
      (testing "max pool backward"
        (is (m/equals [0.0 0.0 0.0 0.0 0.0 1.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0 0.0 0.0
                       0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 2.0 0.0 2.0 0.0 0.0 0.0 0.0
                       0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0
                       0.0 0.0 0.0 1.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 1.0
                       0.0 0.0 0.0 0.0 0.0 1.0 0.0 1.0 0.0 1.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0
                       0.0 1.0 0.0 0.0 0.0 0.0 0.0 2.0 0.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0]
                      (vec (ct/to-double-array max-input-gradient))
                      1e-4)))
      (testing "avg pool backward"
        (is (m/equals [0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111
                       0.2222222222222222 0.4444444444444444 0.2222222222222222 0.2222222222222222
                       0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111
                       0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111
                       0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111
                       0.2222222222222222 0.4444444444444444 0.2222222222222222 0.2222222222222222
                       0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111
                       0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111
                       0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111
                       0.2222222222222222 0.4444444444444444 0.2222222222222222 0.2222222222222222
                       0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111
                       0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111
                       0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111
                       0.2222222222222222 0.4444444444444444 0.2222222222222222 0.2222222222222222
                       0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111
                       0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111
                       0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111
                       0.2222222222222222 0.4444444444444444 0.2222222222222222 0.2222222222222222
                       0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111
                       0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111
                       0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111
                       0.2222222222222222 0.4444444444444444 0.2222222222222222 0.2222222222222222
                       0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111
                       0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111
                       0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111
                       0.2222222222222222 0.4444444444444444 0.2222222222222222 0.2222222222222222
                       0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111
                       0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111
                       0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111
                       0.2222222222222222 0.4444444444444444 0.2222222222222222 0.2222222222222222
                       0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111
                       0.1111111111111111 0.2222222222222222 0.1111111111111111 0.1111111111111111]
                      (vec (ct/to-double-array avg-input-gradient))
                      1e-4)))
      (testing "avg-exc-pad pool backward"
        (is (m/equals [0.25 0.41666666666666663 0.16666666666666666 0.16666666666666666 0.41666666666666663
                       0.6944444444444444 0.2777777777777778 0.2777777777777778 0.16666666666666666 0.2777777777777778
                       0.1111111111111111 0.1111111111111111 0.16666666666666666 0.2777777777777778 0.1111111111111111
                       0.1111111111111111 0.25 0.41666666666666663 0.16666666666666666 0.16666666666666666
                       0.41666666666666663 0.6944444444444444 0.2777777777777778 0.2777777777777778 0.16666666666666666
                       0.2777777777777778 0.1111111111111111 0.1111111111111111 0.16666666666666666 0.2777777777777778
                       0.1111111111111111 0.1111111111111111 0.25 0.41666666666666663 0.16666666666666666
                       0.16666666666666666 0.41666666666666663 0.6944444444444444 0.2777777777777778 0.2777777777777778
                       0.16666666666666666 0.2777777777777778 0.1111111111111111 0.1111111111111111 0.16666666666666666
                       0.2777777777777778 0.1111111111111111 0.1111111111111111 0.25 0.41666666666666663
                       0.16666666666666666 0.16666666666666666 0.41666666666666663 0.6944444444444444 0.2777777777777778
                       0.2777777777777778 0.16666666666666666 0.2777777777777778 0.1111111111111111 0.1111111111111111
                       0.16666666666666666 0.2777777777777778 0.1111111111111111 0.1111111111111111 0.25
                       0.41666666666666663 0.16666666666666666 0.16666666666666666 0.41666666666666663 0.6944444444444444
                       0.2777777777777778 0.2777777777777778 0.16666666666666666 0.2777777777777778 0.1111111111111111
                       0.1111111111111111 0.16666666666666666 0.2777777777777778 0.1111111111111111 0.1111111111111111
                       0.25 0.41666666666666663 0.16666666666666666 0.16666666666666666 0.41666666666666663
                       0.6944444444444444 0.2777777777777778 0.2777777777777778 0.16666666666666666 0.2777777777777778
                       0.1111111111111111 0.1111111111111111 0.16666666666666666 0.2777777777777778 0.1111111111111111
                       0.1111111111111111 0.25 0.41666666666666663 0.16666666666666666 0.16666666666666666
                       0.41666666666666663 0.6944444444444444 0.2777777777777778 0.2777777777777778 0.16666666666666666
                       0.2777777777777778 0.1111111111111111 0.1111111111111111 0.16666666666666666 0.2777777777777778
                       0.1111111111111111 0.1111111111111111 0.25 0.41666666666666663 0.16666666666666666
                       0.16666666666666666 0.41666666666666663 0.6944444444444444 0.2777777777777778 0.2777777777777778
                       0.16666666666666666 0.2777777777777778 0.1111111111111111 0.1111111111111111 0.16666666666666666
                       0.2777777777777778 0.1111111111111111 0.1111111111111111]
                      (vec (ct/to-double-array avg-exc-pad-input-gradient))
                      1e-4))))))

(defn rand-operator
  [driver datatype]
  (tensor-context
   driver datatype
   (testing "Gaussian rand"
     (let [test-vec (->> (range 1 11)
                         (mapcat (fn [idx]
                                   (let [tens (tensor/rand! (tensor/new-tensor [10000])
                                                            (tensor/gaussian-distribution
                                                             :mean idx :variance idx))
                                         values (tensor/to-double-array tens)]
                                     [(stats/mean values)
                                      (stats/variance values)])))
                         vec)]
       (is (m/equals [1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10]
                     test-vec
                     1))))
   (testing "Flat rand"
     (let [test-vec (->> (range 1 11)
                         (mapcat (fn [idx]
                                   (let [tens (tensor/rand! (tensor/new-tensor [10000])
                                                            (tensor/flat-distribution
                                                             :minimum (- idx 2)
                                                             :maximum (+ idx 2)))
                                         values (tensor/to-double-array tens)]
                                     [(stats/mean values)])))
                         vec)]
       (is (m/equals [1 2 3 4 5 6 7 8 9 10]
                     test-vec
                     1))))))


(defn- lrn-forward-backward
  [num-input-channels lrn-n]
   (let [batch-size 2
         input-dim 2
         input-num-pixels (* input-dim input-dim)
         n-input (* num-input-channels input-num-pixels)
         input (-> (tensor/->tensor (flatten (repeat batch-size (range n-input))))
                   (tensor/in-place-reshape [batch-size num-input-channels input-dim input-dim]))
         output-gradient (-> (tensor/->tensor (repeat (* batch-size n-input) 1.0))
                             (tensor/in-place-reshape [batch-size num-input-channels input-dim input-dim]))
         output (tensor/new-tensor (m/shape input))
         input-gradient (tensor/new-tensor (m/shape input))
         lrn-desc (tensor/lrn-descriptor :n lrn-n :k 1 :alpha 1 :beta 1)]
     (tensor/lrn-forward! output input lrn-desc)
     (tensor/lrn-backward! input-gradient output input output-gradient lrn-desc)
     {:output (vec (tensor/to-double-array output))
      :input-gradient (vec (tensor/to-double-array input-gradient))
      :input-data (vec (tensor/to-double-array input))}))


(defn lrn-operator
  [driver datatype]
  (tensor-context
   driver datatype
   (testing "lrn with n 1"
     (let [{:keys [output input-gradient input-data]} (lrn-forward-backward 3 1)]
       (is (m/equals (mapv #(/ (double %) (+ 1 (* % %))) input-data)
                     output
                     1e-4))
       (is (m/equals [1.0 0.0 -0.12000000000000002 -0.07999999999999999 -0.0519031141868512
                      -0.03550295857988165 -0.02556610664718773 -0.019200000000000002
                      -0.0149112426035503  -0.01189767995240928 -0.009704930889128518
                      -0.008062348830959416 1.0 0.0 -0.12000000000000002 -0.07999999999999999
                      -0.0519031141868512 -0.03550295857988165 -0.02556610664718773 -0.019200000000000002
                      -0.0149112426035503 -0.01189767995240928 -0.009704930889128518 -0.008062348830959416]
                     input-gradient
                     1e-4))))
   (testing "lrn with n 2"
     (let [{:keys [output input-gradient]} (lrn-forward-backward 3 2)]
       (is (m/equals (vec
                      (flatten
                       (repeat 2
                               [0.0 0.07142857142857142 0.09523809523809523 0.1
                                0.0975609756097561 0.09259259259259259 0.08695652173913043
                                0.08139534883720931 0.24242424242424243 0.21686746987951808
                                0.19607843137254902 0.17886178861788618])))
                     output
                     1e-4))
       (is (m/equals [0.1111111111111111 0.06461185297164132 0.03602827394347783 0.020493960699477193
                      -0.014512656716972328 -0.016183480513356906 -0.016136733799491102 -0.015355548198598244
                      -0.02846648301193755 -0.022935113949774988 -0.0188389081122645 -0.015731376825963386
                      0.1111111111111111 0.06461185297164132 0.03602827394347783 0.020493960699477193
                      -0.014512656716972328 -0.016183480513356906 -0.016136733799491102 -0.015355548198598244
                      -0.02846648301193755 -0.022935113949774988 -0.0188389081122645 -0.015731376825963386]
                     input-gradient
                     1e-4))))
   (testing "lrn with n 3"
     (let [{:keys [output input-gradient]} (lrn-forward-backward 3 3)]
      (is (m/equals (mapv double
                          (flatten
                           (repeat 2
                                   [0.0 0.10344827586206898 0.13953488372093023
                                    0.14754098360655737 0.14457831325301207 0.13636363636363638
                                    0.1258741258741259 0.11538461538461539 0.28915662650602414
                                    0.24770642201834867 0.21582733812949642
                                    0.19075144508670522 ])))
                    output
                    1e-4))
      (is (m/equals  [0.15789473684210528 0.09383457316653729 0.05326649810721716 0.030864211553852466
                      -0.005661199012919144 -0.04352114602312829 -0.04715638623434852 -0.04169062137988738
                      -0.047466976339091305 -0.03569676147971518 -0.027076332292621106 -0.020863959610017586
                      0.15789473684210528 0.09383457316653729 0.05326649810721716 0.030864211553852466
                      -0.005661199012919144 -0.04352114602312829 -0.04715638623434852 -0.04169062137988738
                      -0.047466976339091305 -0.03569676147971518 -0.027076332292621106 -0.020863959610017586]
                    input-gradient
                    1e-4))))))
