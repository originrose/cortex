(ns think.compute.array-view-math
  (:require [think.compute.math-util :refer :all]
            [clojure.core.matrix.macros :refer [c-for]]
            [think.datatype.core :refer [v-aget-rem v-aset-rem v-aget v-aset] :as dtype]
            [think.resource.core :as resource])
  (:import [com.github.fommil.netlib BLAS]
           [java.util Random]
           [think.datatype DoubleArrayView FloatArrayView ArrayView
            LongArrayView IntArrayView ShortArrayView ByteArrayView]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defprotocol PCPUMathImpl
  (gemm [A a-colstride
             trans-a? trans-b? a-row-count a-col-count b-col-count alpha
             B b-colstride
             beta C c-colstride])
  (sum [x alpha beta y result])
  (gemv [A a-colstride trans-a a-row-count a-col-count alpha x inc-x beta y inc-y])
  (mul-rows [A a-colstride a-row-count a-col-count X inc-x C c-colstride])
  (elem-mul [a inc-a alpha b inc-b res inc-res])
  ;;Create a scale vector with either 1.0 in the row if the row-len is < the
  ;;l2 constraint or (/ l2-max-constraint row-len) otherwise.
  (l2-constraint-scale [a inc-a l2-max-constraint])
  (generate-rands [rand-buffer distribution]))


(defmacro sum-impl
  [x alpha beta y result cast-fn]
  `(let [alpha# (~cast-fn ~alpha)
         beta# (~cast-fn ~beta)
         y-view# (ArrayView/toView ~y)
         x-view# (ArrayView/toView ~x)
         res-view# (ArrayView/toView ~result)
         num-elems# (Math/max (.length x-view#) (.length y-view#))]
     (c-for [idx# 0 (< idx# num-elems#) (inc idx#)]
            (v-aset-rem res-view# idx#
                  (+ (* alpha# (v-aget-rem x-view# idx#))
                     (* beta# (v-aget-rem y-view# idx#)))))))


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
  (gemm [^DoubleArrayView A a-colstride
             trans-a? trans-b? a-row-count a-col-count b-col-count alpha
             ^DoubleArrayView B b-colstride
             beta ^DoubleArrayView C c-colstride]
    (col->row-gemm (fn [trans-a? trans-b? a-row-count a-col-count b-col-count
                        alpha ^DoubleArrayView A a-rowstride
                        ^DoubleArrayView B b-rowstride
                        beta ^DoubleArrayView C c-rowstride]
                     (let [trans-a? (bool->blas-trans trans-a?)
                           trans-b? (bool->blas-trans trans-b?)
                           M (long a-row-count)
                           N (long b-col-count)
                           K (long a-col-count)
                           alpha (double alpha)
                           beta (double beta)
                           A-offset (.offset A)
                           B-offset (.offset B)
                           C-offset (.offset C)
                           A (.data A)
                           B (.data B)
                           C (.data C)]
                       (.dgemm (BLAS/getInstance) trans-a? trans-b?
                               M N K
                               alpha A A-offset a-rowstride
                               B B-offset b-rowstride
                               beta C C-offset c-rowstride)))
                   trans-a? trans-b? a-row-count a-col-count b-col-count
                   alpha A a-colstride
                   B b-colstride
                   beta C c-colstride))
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
  (generate-rands [^DoubleArrayView rand-buffer distribution elem-count]
    (throw (Exception. "Random generation operates on float buffers for CUDA compatibility")))

  FloatArrayView
  (gemm [^FloatArrayView A a-colstride
             trans-a? trans-b? a-row-count a-col-count b-col-count alpha
             ^FloatArrayView B b-colstride
             beta ^FloatArrayView C c-colstride]
    (col->row-gemm (fn [trans-a? trans-b? a-row-count a-col-count b-col-count
                        alpha ^FloatArrayView A a-rowstride
                        ^FloatArrayView B b-rowstride
                        beta ^FloatArrayView C c-rowstride]
                     (let [trans-a? (bool->blas-trans trans-a?)
                           trans-b? (bool->blas-trans trans-b?)
                           M (long a-row-count)
                           N (long b-col-count)
                           K (long a-col-count)
                           alpha (float alpha)
                           beta (float beta)
                           A-offset (.offset A)
                           B-offset (.offset B)
                           C-offset (.offset C)
                           A (.data A)
                           B (.data B)
                           C (.data C)]
                       (.sgemm (BLAS/getInstance) trans-a? trans-b?
                               M N K
                               alpha A A-offset a-rowstride
                               B B-offset b-rowstride
                               beta C C-offset c-rowstride)))
                   trans-a? trans-b? a-row-count a-col-count b-col-count
                   alpha A a-colstride
                   B b-colstride
                   beta C c-colstride))
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
    (l2-constraint-scale-impl a inc-a l2-max-constraint float))
  (generate-rands [^FloatArrayView rand-buffer distribution]
    (let [rand-view (ArrayView/toView rand-buffer)
          rand-gen (Random.)
          elem-count (.length rand-view)]
      (cond
        (= (:type distribution) :gaussian)
        (let [mean (float (:mean distribution))
              variance (float (:variance distribution))
              sum-var (float-array 2)]
          (c-for [idx 0 (< idx elem-count) (inc idx)]
                 (let [next-rand (.nextGaussian rand-gen)]
                   (v-aset rand-view idx next-rand)
                   (aset sum-var 0 (+ (aget sum-var 0) next-rand))
                   (aset sum-var 1 (+ (aget sum-var 1)
                                      (float (Math/abs next-rand))))))
          (let [actual-variance (/ (aget sum-var 1) elem-count)
                variance-fix (float (Math/sqrt (if (> actual-variance 0.0)
                                                 (/ variance actual-variance)
                                                 actual-variance)))
                actual-mean (/ (aget sum-var 0) elem-count)
                adjusted-mean (* actual-mean variance-fix)
                mean-fix (- mean adjusted-mean)]
            (c-for [idx 0 (< idx elem-count) (inc idx)]
                   (v-aset rand-view idx (+ (* variance-fix (v-aget rand-view idx))
                                            mean-fix)))))
        (= (:type distribution) :flat)
        (c-for [idx 0 (< idx elem-count) (inc idx)]
               (v-aset rand-view idx (float (.nextFloat rand-gen))))
        :else
        (throw (Exception. (str "Unrecognized distribution: " distribution)))))))


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
