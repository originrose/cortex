(ns cortex.compute.array-view-math
  (:require [cortex.compute.math-util :refer :all]
            [clojure.core.matrix.macros :refer [c-for]]
            [think.datatype.core :refer [v-aget v-aset] :as dtype]
            [think.resource.core :as resource])
  (:import [com.github.fommil.netlib BLAS]
           [java.util Random]
           [think.datatype DoubleArrayView FloatArrayView ArrayView
            LongArrayView IntArrayView ShortArrayView ByteArrayView]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defprotocol PCPUMathImpl
  (sum [x alpha beta y result])
  (gemv [A a-colstride trans-a a-row-count a-col-count alpha x inc-x beta y inc-y])
  (mul-rows [A a-colstride a-row-count a-col-count X inc-x C c-colstride])
  (elem-mul [a inc-a alpha b inc-b res inc-res])
  ;;Create a scale vector with either 1.0 in the row if the row-len is < the
  ;;l2 constraint or (/ l2-max-constraint row-len) otherwise.
  (l2-constraint-scale [a inc-a l2-max-constraint]))


(defmacro sum-impl
  [x alpha beta y result cast-fn]
  `(let [alpha# (~cast-fn ~alpha)
         beta# (~cast-fn ~beta)
         y-view# (ArrayView/toView ~y)
         x-view# (ArrayView/toView ~x)
         res-view# (ArrayView/toView ~result)
         res-len# (.length res-view#)
         x-len# (.length x-view#)
         y-len# (.length y-view#)
         num-elems# (Math/max (.length x-view#) (.length y-view#))]
     (c-for [idx# 0 (< idx# num-elems#) (inc idx#)]
            (v-aset res-view# (rem idx#
                                   res-len#)
                  (+ (* alpha# (v-aget x-view# (rem idx# x-len#)))
                     (* beta# (v-aget y-view# (rem idx# y-len#))))))))


(defmacro mul-rows-impl
  [A a-colstride a-row-count a-col-count x inc-x C c-colstride]
  `(let [~a-colstride (long ~a-colstride)
         ~a-row-count (long ~a-row-count)
         ~a-col-count (long ~a-col-count)
         ~inc-x (long ~inc-x)
         ~c-colstride (long ~c-colstride)
         A# (ArrayView/toView ~A)
         x# (ArrayView/toView ~x)
         C# (ArrayView/toView ~C)]
     (c-for
      [row# 0 (< row# ~a-row-count) (inc row#)]
      (let [a-row-offset# (* ~a-colstride row#)
            x-offset# (* row# ~inc-x)
            c-row-offset# (* ~c-colstride row#)
            x-val# (v-aget x# x-offset#)]
        (c-for
         [col# 0 (< col# ~a-col-count) (inc col#)]
         (v-aset C# (+ c-row-offset# col#)
               (* x-val# (v-aget A# (+ a-row-offset# col#)))))))))

(defmacro elem-mul-impl
  [a inc-a alpha b inc-b res inc-res cast-fn]
  `(let [alpha# (~cast-fn ~alpha)
         a# (ArrayView/toView ~a)
         inc-a# (long ~inc-a)
         b# (ArrayView/toView ~b)
         inc-b# (long ~inc-b)
         res# (ArrayView/toView ~res)
         inc-res# (long ~inc-res)
         elem-count# (quot (.length a#) inc-a#)]
     (c-for [idx# 0 (< idx# elem-count#) (inc idx#)]
            (v-aset res# (* inc-res# idx#)
                    (* (* alpha# (v-aget a# (* inc-a# idx#)))
                       (v-aget b# (* inc-b# idx#)))))))

(defmacro l2-constraint-scale-impl
  [a inc-a l2-max-constraint cast-fn]
  `(let [a# (ArrayView/toView ~a)
         inc-a# (long ~inc-a)
         a-elem-count# (quot (.length a#) inc-a#)
         l2-max-constraint# (~cast-fn ~l2-max-constraint)]
     (c-for [idx# 0 (< idx# a-elem-count#) (inc idx#)]
            (let [a-offset# (* idx# inc-a#)
                  row-len# (Math/sqrt (v-aget a# a-offset#))]
              (if (< row-len# l2-max-constraint#)
                (v-aset a# a-offset# 1.0)
                (v-aset a# a-offset# (/ l2-max-constraint# row-len#)))))))


(extend-protocol PCPUMathImpl
  DoubleArrayView
  (sum [^DoubleArrayView x alpha beta
            ^DoubleArrayView y
            ^DoubleArrayView result]
    (sum-impl x alpha beta y result double))
  (gemv [^DoubleArrayView A a-colstride trans-a? a-row-count a-col-count
             alpha ^DoubleArrayView x inc-x
             beta ^DoubleArrayView y inc-y]
    (col->row-gemv (fn [trans-a? a-row-count a-col-count
                        alpha ^DoubleArrayView A a-colstride
                        ^DoubleArrayView x inc-x
                        beta ^DoubleArrayView y inc-y]
                     (let [a-colstride (long a-colstride)
                           a-row-count (long a-row-count)
                           a-col-count (long a-col-count)
                           A-offset (.offset A)
                           x-offset (.offset x)
                           y-offset (.offset y)
                           A (.data A)
                           x (.data x)
                           y (.data y)
                           alpha (double alpha)
                           inc-x (long inc-x)
                           beta (double beta)
                           inc-y (long inc-y)]
                       (.dgemv (BLAS/getInstance) (bool->blas-trans trans-a?)
                               a-row-count a-col-count
                               alpha A A-offset a-colstride
                               x x-offset inc-x
                               beta y y-offset inc-y)))
                   trans-a? a-row-count a-col-count
                   alpha A a-colstride
                   x inc-x
                   beta y inc-y))
  (mul-rows [^DoubleArrayView A a-colstride a-row-count a-col-count
                 ^DoubleArrayView x inc-x ^DoubleArrayView C c-colstride]
    (mul-rows-impl A a-colstride a-row-count a-col-count x inc-x C c-colstride))
  (elem-mul [^DoubleArrayView a inc-a alpha ^DoubleArrayView b inc-b
                 ^DoubleArrayView res inc-res]
    (elem-mul-impl a inc-a alpha b inc-b res inc-res double))
  (l2-constraint-scale [^DoubleArrayView a inc-a l2-max-constraint]
    (l2-constraint-scale-impl a inc-a l2-max-constraint double))

  FloatArrayView
  (sum [^FloatArrayView x alpha beta
            ^FloatArrayView y
            ^FloatArrayView result]
    (sum-impl x alpha beta y result float))
  (gemv [^FloatArrayView A a-colstride trans-a? a-row-count a-col-count
             alpha ^FloatArrayView x inc-x
             beta ^FloatArrayView y inc-y]
    (col->row-gemv (fn [trans-a? a-row-count a-col-count
                        alpha ^FloatArrayView A a-colstride
                        ^FloatArrayView x inc-x
                        beta ^FloatArrayView y inc-y]
                     (let [a-colstride (long a-colstride)
                           a-row-count (long a-row-count)
                           a-col-count (long a-col-count)
                           A-offset (.offset A)
                           x-offset (.offset x)
                           y-offset (.offset y)
                           A (.data A)
                           x (.data x)
                           y (.data y)
                           alpha (float alpha)
                           inc-x (long inc-x)
                           beta (float beta)
                           inc-y (long inc-y)]
                       (.sgemv (BLAS/getInstance) (bool->blas-trans trans-a?)
                               a-row-count a-col-count
                               alpha A A-offset a-colstride
                               x x-offset inc-x
                               beta y y-offset inc-y)))
                   trans-a? a-row-count a-col-count
                   alpha A a-colstride
                   x inc-x
                   beta y inc-y))
  (mul-rows [^FloatArrayView A a-colstride a-row-count a-col-count
                 ^FloatArrayView x inc-x ^FloatArrayView C c-colstride]
    (mul-rows-impl A a-colstride a-row-count a-col-count x inc-x C c-colstride))
  (elem-mul [^FloatArrayView a inc-a alpha ^FloatArrayView b inc-b
                 ^FloatArrayView res inc-res]
    (elem-mul-impl a inc-a alpha b inc-b res inc-res float))
  (l2-constraint-scale [^FloatArrayView a inc-a l2-max-constraint]
    (l2-constraint-scale-impl a inc-a l2-max-constraint float)))


(extend-protocol resource/PResource
  ByteArrayView
  (release-resource [item])
  ShortArrayView
  (release-resource [item])
  IntArrayView
  (release-resource [item])
  LongArrayView
  (release-resource [item])
  FloatArrayView
  (release-resource [item])
  DoubleArrayView
  (release-resource [item]))
