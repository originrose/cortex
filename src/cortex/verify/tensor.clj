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
