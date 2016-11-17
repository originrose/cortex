(ns think.compute.nn.cpu-backend
  (:require [think.compute.nn.backend :as nn-backend]
            [think.compute.driver :as drv]
            [think.compute.math :as math]
            [think.compute.cpu-driver :as cpu-drv]
            [think.datatype.core :refer [v-aget v-aset v-alength] :as dtype]
            [think.compute.optimise :as opt]
            [think.compute.nn.layers :as layers]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.protocols :as mp]
            [clojure.core.matrix.macros :refer [c-for]]
            [cortex.nn.impl.layers.convolution :as conv])
  (:import [think.compute.cpu_driver CPUDriver CPUStream]
           [java.nio DoubleBuffer FloatBuffer]
           [think.compute.math DeviceArray Tensor]
           [think.compute.optimise AdadeltaOptimiser AdamOptimiser]
           [cortex.nn.impl.layers.convolution ConvLayerConfig]
           [java.util Arrays]
           [java.util.concurrent ForkJoinPool Callable Future]
           [think.datatype ArrayView DoubleArrayView FloatArrayView]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defrecord CPUBackend [^CPUDriver driver ^CPUStream stream datatype])


(defn create-cpu-backend
  (^CPUDriver [datatype]
   (let [driver (cpu-drv/create-driver)
         stream (drv/create-stream driver)]
     (->CPUBackend driver stream datatype)))
  (^CPUBackend []
   (create-cpu-backend :double)))


(defprotocol PCPUNetworkImpl
  "Implementation of various functions based on buffer datatype."
  (cpu-activation-forward [input-buf act-type output-buf])
  (cpu-activation-backward [input-buf act-type output-buf
                            output-gradient input-gradient])
  (cpu-softmax-forward [input-buf output-buf n-input n-channels])
  (cpu-adadelta-step! [gradient parameters gradient-alpha param-offset
                       decay epsilon grad-accum dx-accum])
  (cpu-adam-step! [gradient parameters gradient-alpha param-offset alpha beta1 beta2 epsilon
                   pow_beta1_t pow_beta2_t m v])
  (cpu-planar-input->convolution! [input input-convolved conv-config])
  (cpu-convolution->planar-output! [input-convolved input-gradient conv-config])
  (cpu-fill [buffer value])
  (cpu-max-pooling-forward [input output conv-config])
  (cpu-max-pooling-backward [input output input-gradient output-gradient conv-config])
  (cpu-prepare-bernoulli-dropout [mult-buffer rand-buffer probability])
  (cpu-prepare-gaussian-dropout [mult-buffer rand-buffer])
  (cpu-bn-calc [input running-means running-variances
                scale bias output batch-size batch-stride])
  (cpu-update-means-variances [input
                               running-means running-variances
                               saved-means saved-variances
                               batch-size batch-stride ave-factor epsilon])
  (cpu-bn-backward [input saved-means saved-variances scale bias output
                    scale-gradient bias-gradient input-gradient
                    output-gradient batch-size batch-stride])
  (cpu-lrn-forward [input output input-tensor n k alpha beta])
  (cpu-lrn-backward [input output-gradient input-gradient input-tensor n k alpha beta]))

(defn launch-parallel-for
  [^long num-iters parallel-for-fn]
  (if (< num-iters (* 2 (ForkJoinPool/getCommonPoolParallelism)))
    (parallel-for-fn 0 num-iters)
    (let [num-iters (long num-iters)
          parallelism (ForkJoinPool/getCommonPoolParallelism)
          group-size (quot num-iters parallelism)
          overflow (rem num-iters parallelism)
          overflow-size (+ group-size 1)
          group-count (min num-iters parallelism)
          ;;Get pairs of (start-idx, len) to launch callables
          groups (map (fn [^long callable-idx]
                        (let [group-len (if (< callable-idx overflow)
                                          overflow-size
                                          group-size)
                              group-start (+ (* overflow-size
                                                (min overflow callable-idx))
                                             (* group-size
                                                (max 0 (- callable-idx overflow))))]
                          [group-start group-len]))
                      (range parallelism))
          callables (map (fn [[start-idx len]]
                           (reify Callable
                             (call [this]
                               (parallel-for-fn start-idx len))))
                         groups)
          common-pool (ForkJoinPool/commonPool)
          ;;launch the missiles
          futures (mapv (fn [^Callable c]
                          (.submit common-pool c))
                        callables)]
      (doseq [^Future fut futures]
        (.get fut)))))


(defmacro parallel-for
  [idx-var num-iters & body]
  `(launch-parallel-for ~num-iters
                        (fn [^long group-start# ^long group-len#]
                          (let [group-end# (+ group-start# group-len#)]
                            (c-for [~idx-var group-start#
                                    (< ~idx-var group-end#)
                                    (inc ~idx-var)]
                                   ~@body)))))


(defmacro cpu-act-forward-impl
  [act-type input output cast-fn]
  `(let [src# (ArrayView/toView ~input)
         dest# (ArrayView/toView ~output)
         n-elems# (.length src#)
         val-0# (~cast-fn 0)
         val-1# (~cast-fn 1)]
     (cond
       (= ~act-type :sigmoid)
       (c-for [idx# 0 (< idx# n-elems#) (inc idx#)]
              (v-aset dest# idx#
                    (/ val-1#
                       (+ val-1# (Math/exp (- (v-aget src# idx#)))))))
       (= ~act-type :relu)
       (c-for [idx# 0 (< idx# n-elems#) (inc idx#)]
              (v-aset dest# idx#
                    (Math/max (v-aget src# idx#) val-0#)))
       (= ~act-type :tanh)
       (c-for [idx# 0 (< idx# n-elems#) (inc idx#)]
              (v-aset dest# idx#
                    (Math/tanh (v-aget src# idx#)))))))

(defmacro cpu-act-backward-impl
  [act-type input output output-gradient input-gradient cast-fn]
  `(let [src# (ArrayView/toView ~input)
         dest# (ArrayView/toView ~output)
         src-grad# (ArrayView/toView ~input-gradient)
         dest-grad# (ArrayView/toView ~output-gradient)
         n-elems# (.length src#)
         val-1# (~cast-fn 1)
         val-0# (~cast-fn 0)]
     (cond
       (= ~act-type :sigmoid)
       ;; input gradient = output * (1 - output) * output-gradient
       (c-for [idx# 0 (< idx# n-elems#) (inc idx#)]
              (let [out-val# (v-aget dest# idx#)]
                (v-aset src-grad# idx#
                      (* out-val#
                         (- val-1# out-val#)
                         (v-aget dest-grad# idx#)))))
       (= ~act-type :relu)
       (c-for [idx# 0 (< idx# n-elems#) (inc idx#)]
              (let [mult# (~cast-fn (if (> (v-aget src# idx#)
                                           val-0#)
                                      1
                                      0))]
                (v-aset src-grad# idx#
                      (* mult# (v-aget dest-grad# idx#)))))
       (= ~act-type :tanh)
       (c-for [idx# 0 (< idx# n-elems#) (inc idx#)]
              (let [out-val# (v-aget dest# idx#)]
                (v-aset src-grad# idx#
                      (* (- val-1#
                            (* out-val# out-val#))
                         (v-aget dest-grad# idx#))))))))

(defmacro array-max
  [ary n-items start-idx cast-fn]
  `(loop [idx# 1
          max-val# (v-aget ~ary ~start-idx)]
     (if (< idx# ~n-items)
       (recur (inc idx#)
              (Math/max (~cast-fn max-val#) (v-aget ~ary (+ ~start-idx idx#))))
       max-val#)))

(defmacro array-sum
  [ary n-items start-idx]
  `(loop [idx# 1
          sum-val# (v-aget ~ary ~start-idx)]
     (if (< idx# ~n-items)
       (recur (inc idx#)
              (+ sum-val# (v-aget ~ary (+ ~start-idx idx#))))
       sum-val#)))


(defmacro cpu-view-softmax
  [src dest cast-fn]
  `(let [num-items# (.length ~src)
         max-val# (~cast-fn (array-max ~src num-items# 0 ~cast-fn))]
     ;;Subtract max for numerical stability
     (c-for [idx# 0 (< idx# num-items#) (inc idx#)]
            (v-aset ~dest idx# (Math/exp (- (v-aget ~src idx#) max-val#))))
     ;;perform normalization with array sum.
     (let [sum-val# (~cast-fn (array-sum ~dest num-items# 0))]
       (c-for [idx# 0 (< idx# num-items#) (inc idx#)]
              (.diveq ~dest idx# sum-val#)))))



(defmacro cpu-softmax-forward-impl
  [n-input input-buf output-buf n-channels cast-fn]
  `(let [n-input# (long ~n-input)
         src# (ArrayView/toView ~input-buf)
         dest# (ArrayView/toView ~output-buf)
         batch-size# (quot (v-alength src#) n-input#)
         n-channels# (long ~n-channels)]
     (c-for [batch-idx# 0 (< batch-idx# batch-size#) (inc batch-idx#)]
            (if (= n-channels# 1)
              (let [start-offset# (* batch-idx# n-input#)
                    max-val# (~cast-fn (array-max src# n-input# start-offset# ~cast-fn))
                    src# (.toView src# start-offset# n-input#)
                    dest# (.toView dest# start-offset# n-input#)]
                (cpu-view-softmax src# dest# ~cast-fn))
              (let [start-offset# (* batch-idx# n-input#)
                    n-pixels# (quot n-input# n-channels#)]
                (parallel-for
                 pixel# n-pixels#
                 (cpu-view-softmax (.toView src# (+ start-offset#
                                                    (* pixel# n-channels#))
                                            n-channels#)
                                   (.toView dest# (+ start-offset#
                                                     (* pixel# n-channels#))
                                            n-channels#)
                                   ~cast-fn)))))))



(defmacro cpu-planar-input->convolution!-impl
  [input output config cast-fn]
  `(let [input-ary# (ArrayView/toView ~input)
         output-ary# (ArrayView/toView ~output)]
     (conv/convolution-outer-kernel
      ~config
      :convolutional
      (conv/convolution-roll-unroll-inner-kernel
       (let [input-val# (~cast-fn (if ~'input-valid?
                                    (v-aget input-ary# ~'input-addr)
                                    0.0))]
         (v-aset output-ary# ~'output-conv-addr input-val#))))))


(defmacro cpu-convolution->planar-output!-impl
  "Sum the convolution up to the planar input."
  [conv-input-gradient input-gradient config cast-fn]
  ;;I am using input to mean upstream or in this case destination so that
  ;;this code can look as similar to the code above as possible
  ;;This function is extremely confusing but the macros name local variables
  ;;a certain way so in this case input-addr means output-addr.
  `(let [output-ary# (ArrayView/toView ~input-gradient)
         input-ary# (ArrayView/toView ~conv-input-gradient)]
     ;;Zero accumulator
     (conv/convolution-outer-kernel
      ~config :convolutional
      (conv/convolution-roll-unroll-inner-kernel
       (when ~'input-valid?
         (let [output-val# (v-aget output-ary# ~'input-addr)
               input-val# (v-aget input-ary# ~'output-conv-addr)]
           (v-aset output-ary# ~'input-addr (+ input-val# output-val#))))))))

(defmacro cpu-max-pooling-forward-impl
  [input output config cast-fn]
  `(let [input-ary# (ArrayView/toView ~input)
         output-ary# (ArrayView/toView ~output)]
     (conv/convolution-outer-kernel
      ~config :pooling
      (conv/convolution-roll-unroll-inner-kernel
       (let [input-val# (~cast-fn (if ~'input-valid?
                                  (v-aget input-ary# ~'input-addr)
                                  0.0))
             output-addr# (+ (* ~'out-y ~'output-width)
                             ~'out-x
                             ~'chan-output-offset)
             k-idx# (+ (* ~'k-y ~'k-width) ~'k-x)
             output-val# (v-aget output-ary# output-addr#)]
         (when (or (= 0 k-idx#)
                   (> input-val# output-val#))
           (v-aset output-ary# output-addr# input-val#)))))))


(defmacro cpu-max-pooling-backward-impl
  [input output input-gradient output-gradient config cast-fn]
  `(let [input-ary# (ArrayView/toView ~input)
         output-ary# (ArrayView/toView ~output)
         input-gradient-ary# (ArrayView/toView ~input-gradient)
         output-gradient-ary# (ArrayView/toView ~output-gradient)]
     (conv/convolution-outer-kernel
      ~config :pooling
      (conv/convolution-roll-unroll-inner-kernel
       (let [input-addr# ~'input-addr
             input-val# (~cast-fn (if ~'input-valid?
                                  (v-aget input-ary# input-addr#)
                                  0.0))
             output-addr# (+ (* ~'out-y ~'output-width)
                             ~'out-x
                             ~'chan-output-offset)
             k-idx# (+ (* ~'k-y ~'k-width) ~'k-x)
             output-val# (v-aget output-ary# output-addr#)]
         (when (= input-val# output-val#)
           (v-aset input-gradient-ary# input-addr#
                 (+ (v-aget input-gradient-ary# input-addr#)
                    (v-aget output-gradient-ary# output-addr#)))))))))

(defmacro cpu-prepare-bernoulli-impl
  [mult-buffer rand-buffer probability cast-fn]
  `(let [probability# (~cast-fn ~probability)
         scale-val# (~cast-fn (/ 1.0 probability#))
         mult-ary# (ArrayView/toView ~mult-buffer)
         elem-count# (.length mult-ary#)
         rand-ary# (ArrayView/toView ~rand-buffer)]
     (c-for [idx# 0 (< idx# elem-count#) (inc idx#)]
            (v-aset mult-ary# idx#
                    (~cast-fn (if (> (v-aget rand-ary# idx#)
                                     probability#)
                                0.0
                                scale-val#))))))


(defmacro cpu-prepare-gaussian-impl
  [mult-buffer rand-buffer cast-fn]
  `(let [mult-ary# (ArrayView/toView ~mult-buffer)
         elem-count# (.length mult-ary#)
         rand-ary# (ArrayView/toView ~rand-buffer)]
     (c-for [idx# 0 (< idx# elem-count#) (inc idx#)]
            (v-aset mult-ary# idx#
                    (~cast-fn (v-aget rand-ary# idx#))))))


(defmacro cpu-bn-calc-impl
  [input means variances scale bias output batch-size batch-stride cast-fn]
  `(let [input-ary# (ArrayView/toView ~input)
         means-ary# (ArrayView/toView ~means)
         variances-ary# (ArrayView/toView ~variances)
         scale-ary# (ArrayView/toView ~scale)
         bias-ary# (ArrayView/toView ~bias)
         output-ary# (ArrayView/toView ~output)
         batch-size# (long ~batch-size)
         batch-stride# (long ~batch-stride)]
     (parallel-for
      elem-idx# batch-stride#
      (let [inv-std-dev# (Math/sqrt (/ 1.0
                                       (v-aget variances-ary# elem-idx#)))
            mean# (v-aget means-ary# elem-idx#)
            scale# (v-aget scale-ary# elem-idx#)
            shift# (v-aget bias-ary# elem-idx#)]
        (c-for
         [batch-idx# 0 (< batch-idx# batch-size#) (inc batch-idx#)]
         (let [item-offset# (+ (* batch-idx# batch-stride#) elem-idx#)
               x-hat# (* (- (v-aget input-ary# item-offset#) mean#)
                         inv-std-dev#)]
           (v-aset output-ary# item-offset#
                   (+ (* x-hat# scale#) shift#))))))))

(defmacro sum-double-var
  "Carefully written macro to sum a double variable.  Note that we are careful
to avoid adding the first calculated answer to 0.0 as if that answer is very small
we would introduce roundoff error immediately.  So we need a slightly more complex loop
in order to avoid adding a small number to 0."
  [idx-var num-iters stmt]
  `(double
    (if (= 0 ~num-iters)
        0.0
        (loop [sum-var# (let [~idx-var 0] ~stmt)
               ~idx-var 1]
          (if (< ~idx-var ~num-iters)
            (recur (+ sum-var# ~stmt) (inc ~idx-var))
            sum-var#)))))

(defmacro cpu-update-means-variances-impl
  [input running-means running-variances
   saved-means saved-variances
   batch-size batch-stride ave-factor epsilon cast-fn]
  `(let [input-ary# (ArrayView/toView ~input)
         running-means-ary# (ArrayView/toView ~running-means)
         running-variances-ary# (ArrayView/toView ~running-variances)
         saved-means-ary# (ArrayView/toView ~saved-means)
         saved-variances-ary# (ArrayView/toView ~saved-variances)
         batch-size# (long ~batch-size)
         batch-stride# (long ~batch-stride)
         ave-factor# (~cast-fn ~ave-factor)
         ave-lerp# (- (~cast-fn 1.0) ave-factor#)
         epsilon# (~cast-fn ~epsilon)]
     (parallel-for elem-idx# batch-stride#
      (let [variance# (v-aget running-variances-ary# elem-idx#)
            mean# (v-aget running-means-ary# elem-idx#)
            input-idx# elem-idx#
            batch-size-val# (double batch-size#)
            var-batch-size# (max 1.0 (- batch-size-val# 1.0))
            new-mean# (~cast-fn
                       (/ (sum-double-var batch-idx# batch-size#
                                          (v-aget input-ary#
                                                  (+ input-idx#
                                                     (* batch-idx# batch-stride#))))
                          batch-size-val#))

            new-var# (double
                      (+
                       epsilon#
                       (sum-double-var batch-idx# batch-size#
                                       (let [mean-diff# (- new-mean#
                                                           (v-aget input-ary#
                                                                   (+ input-idx#
                                                                      (* batch-idx#
                                                                         batch-stride#))))]
                                         (* mean-diff# mean-diff#)))))]
        (v-aset saved-means-ary# elem-idx# new-mean#)
        (v-aset saved-variances-ary# elem-idx# (~cast-fn
                                                (/ new-var#
                                                   batch-size-val#)))
        (v-aset running-means-ary# elem-idx#
                (+ (* mean# ave-lerp#) (* new-mean# ave-factor#)))
        (v-aset running-variances-ary# elem-idx#
                (+ (* variance# ave-lerp#) (* (~cast-fn (/ new-var#
                                                           var-batch-size#))
                                              ave-factor#)))))))

(defmacro cpu-bn-backward-impl [input means variances scale bias output
                                scale-gradient bias-gradient input-gradient
                                output-gradient batch-size batch-stride cast-fn]
  `(let [batch-size# (long ~batch-size)
         batch-stride# (long ~batch-stride)
         pow-factor# (~cast-fn (/ -3.0 2.0))]
     (parallel-for
      elem-idx# batch-stride#
      (let [input-ary# (ArrayView/toView ~input elem-idx#)
            means-ary# (ArrayView/toView ~means elem-idx#)
            variances-ary# (ArrayView/toView ~variances elem-idx#)
            scale-ary# (ArrayView/toView ~scale elem-idx#)
            bias-ary# (ArrayView/toView ~bias elem-idx#)
            output-ary# (ArrayView/toView ~output elem-idx#)
            scale-gradient-ary# (ArrayView/toView ~scale-gradient elem-idx#)
            bias-gradient-ary# (ArrayView/toView ~bias-gradient elem-idx#)
            input-gradient-ary# (ArrayView/toView ~input-gradient elem-idx#)
            output-gradient-ary# (ArrayView/toView ~output-gradient elem-idx#)
            scale# (v-aget scale-ary# 0)
            inv-variance# (/ 1.0
                             (v-aget variances-ary# 0))
            inv-std-dev# (Math/sqrt inv-variance#)
            mean# (v-aget means-ary# 0)
            d-x-hat-d-out-ary# input-gradient-ary#
            d-var# (~cast-fn (* -0.5 (Math/pow (/ 1.0 inv-variance#) pow-factor#)))]
        ;;(println 1)
        ;;These sums are somewhat inefficient but the math is so complicated
        ;;that I want to lay it out without combining loops.
        (v-aset bias-gradient-ary# 0
                (~cast-fn (sum-double-var
                           batch-idx# batch-size#
                           (let [batch-offset# (* batch-idx# batch-stride#)]
                             (v-aget output-gradient-ary# batch-offset#)))))
        ;;(println 2)
        (v-aset scale-gradient-ary# 0
                (~cast-fn (sum-double-var
                           batch-idx# batch-size#
                           (let [batch-offset# (* batch-idx# batch-stride#)]
                             (* (v-aget output-gradient-ary# batch-offset#)
                                (* (- (v-aget input-ary# batch-offset#)
                                      mean#)
                                   inv-std-dev#))))))
        ;;(println 3)
        ;;run through get get d-x-hat/d-output.  Store in input-gradient
        (c-for [batch-idx# 0 (< batch-idx# batch-size#) (inc batch-idx#)]
               (let [batch-offset# (* batch-idx# batch-stride#)]
                 (v-aset d-x-hat-d-out-ary# batch-offset#
                       (* scale# (v-aget output-gradient-ary# batch-offset#)))))
        ;;(println 4)
        ;;Input gradient calculation...
        (let [d-var-d-out# (~cast-fn
                            (sum-double-var
                             batch-idx# batch-size#
                             (let [batch-offset# (* batch-idx# batch-stride#)]
                               (* (v-aget d-x-hat-d-out-ary# batch-offset#)
                                  (- (v-aget input-ary# batch-offset#)
                                     mean#)
                                  d-var#))))
              d-mean-d-out# (~cast-fn
                             (+ (sum-double-var
                                 batch-idx# batch-size#
                                 (let [batch-offset# (* batch-idx# batch-stride#)]
                                   (* (- (v-aget d-x-hat-d-out-ary# batch-offset# ))
                                      inv-std-dev#)))
                                (* d-var-d-out#
                                   (/ (sum-double-var
                                       batch-idx# batch-size#
                                       (let [batch-offset# (* batch-idx# batch-stride#)]
                                         (* -2.0
                                            (- (v-aget input-ary# batch-offset#)
                                               mean#))))
                                      batch-size#))))]
          ;;(println 5)
          ;;final input gradient calculation
          (c-for
           [batch-idx# 0 (< batch-idx# batch-size#) (inc batch-idx#)]
           (let [batch-offset# (* batch-idx# batch-stride#)
                 d-x-hat-d-out# (v-aget d-x-hat-d-out-ary# batch-offset#)
                 input-var# (v-aget input-ary# batch-offset#)
                 one-over-batch-size# (/ 1.0 batch-size#)
                 sum-part-1# (* d-x-hat-d-out# inv-std-dev#)
                 sum-part-2# (* d-var-d-out# 2.0 (- input-var# mean#) one-over-batch-size#)
                 sum-part-3# (* d-mean-d-out# one-over-batch-size#)]
             (comment (when (= 0 elem-idx#)
                        (clojure.pprint/pprint
                         [[:sum-part-1 sum-part-1#]
                          [:sum-part-2 sum-part-2#]
                          [:sum-part-3 sum-part-3#]])))
             (v-aset input-gradient-ary# batch-offset#
                     (~cast-fn (+ (+ sum-part-1# sum-part-3#) sum-part-2#)))))

          ;;(println "backward finished")
          )))))


(defmacro a-pluseq
  [ary idx data]
  `(aset ~ary ~idx
         (+ (aget ~ary ~idx) ~data)))


(defmacro a-minuseq
  [ary idx data]
  `(aset ~ary ~idx
         (- (aget ~ary ~idx) ~data)))

(defmacro v-aget-squared
  [ary idx]
  `(let [data-val# (v-aget ~ary ~idx)]
     (* data-val# data-val#)))

(defn calculate-look-amounts
  [^double n]
  (let [look-behind (+ 1 (Math/floor (/ (- n 1) 2.0)))
        look-ahead (- n look-behind)]
    {:remove-index look-behind
     :add-index look-ahead}))

(defmacro calc-squared-sums!
  "Calculate a running squared sum placing result in squared-sum-ary.
remove-index and add index are defined as one past the beginning
of the range relative to n and the end of the range (inclusive).
Calculates: (sum(x[i]^2)*alpha + K)"
  [input n-items remove-index add-index squared-sum-view alpha k cast-fn ]
  `(let [input# ~input
         n-items# ~n-items
         remove-index# ~remove-index
         add-index# ~add-index
         squared-sum-view# ~squared-sum-view
         alpha# (~cast-fn ~alpha)
         k# (~cast-fn ~k)]
     (.fill squared-sum-view# 0)
     (let [initial-sum#
           (loop [item-idx# 0
                  initial-sum# 0.0]
             (if (< item-idx# add-index#)
               (recur (inc item-idx#)
                      (+ initial-sum#
                         (v-aget-squared input# item-idx#)))
               initial-sum#))]
       (loop [item-idx# 0
              running-sum# (double initial-sum#)]
         (when (< item-idx# n-items#)
           (let [subtract-item# (- item-idx# remove-index#)
                 add-item# (+ item-idx# add-index#)
                 running-sum# (double running-sum#)
                 running-sum# (double
                               (if (>= subtract-item# 0)
                                 (- running-sum#
                                    (v-aget-squared input# subtract-item#))
                                 running-sum#))
                 running-sum# (double
                               (if (< add-item# n-items#)
                                 (+ running-sum#
                                    (v-aget-squared input# add-item#))
                                 running-sum#))]
             (.set squared-sum-view# item-idx# (+ k#
                                                  (* alpha#
                                                     (~cast-fn running-sum#))))
             (recur (inc item-idx#) running-sum#)))))))


(defn safe-positive-pow
  "Assuming X is supposed to be positive, perform a safe pow where we do not let X go to zero."
  ^double [^double x ^double y]
  (if (< y 0.0)
    (Math/pow (Math/max x 1e-6) y)
    (Math/pow x y)))


(defmacro cpu-lrn-forward-impl
  [input output input-tensor n k alpha beta cast-fn]
  `(let [input# (ArrayView/toView ~input)
         output# (ArrayView/toView ~output)
         input-tensor# ~input-tensor
         n-channels# (long (.channel-count input-tensor#))
         height# (long (.height input-tensor#))
         width# (long (.width input-tensor#))
         n# (~cast-fn (max 1.0 (~cast-fn (min (~cast-fn ~n) n-channels#))))
         k# (~cast-fn ~k)
         alpha# (/ (~cast-fn ~alpha)
                   n#)
         beta# (~cast-fn ~beta)
         neg-beta# (- beta#)
         [batch-size# batch-stride#] (math/batch-shape input-tensor#)
         batch-size# (long batch-size#)
         batch-stride# (long batch-stride#)
         channel-stride# (* width# height#)
         num-pixels# channel-stride#
         ;; Normalization window width in elements.
         ;; LRN layer uses a window [center-lookBehind, center+lookAhead],
         ;; where lookBehind = floor ( (lrnN-1)/2), lookAhead
         ;; = lrnN-lookBehind-1. So for n=10, the window is [k-4...k...k+5] with a total of 10
         ;; samples.
         ;; cudnnCreateLRNDescriptor.
         look-data# (calculate-look-amounts n#)
         ;;We use a running sum so when going to the next 'n'
         ;;we want to remove the item at 'n - remove-index'
         ;;and add the item at 'n + add-index'.  As such remove-index
         ;;is technically one past the beginning of our range while
         ;;add-index is the last possible valid index of our range.
         remove-index# (long (:remove-index look-data#))
         add-index# (long (:add-index look-data#))]
     (c-for
      [batch-idx# 0 (< batch-idx# batch-size#) (inc batch-idx#)]
      (let [start-offset# (* batch-idx# batch-stride# )
            ;;Create a function that will be used to parallelize the computation
            pixel-fn#
            (fn [^long start# ^long len#]
              (try
               (let [end# (+ start# len#)
                     squared-sum-view# (ArrayView/toView (double-array n-channels#))]
                 (c-for
                  [pix-idx# start# (< pix-idx# end#) (inc pix-idx#)]
                  (let [pix-offset# (+ start-offset# pix-idx#)
                        ;;Setup input to stride over image channels at this pixel.
                        input# (.toStridedView input# pix-offset# channel-stride#)
                        output# (.toStridedView output# pix-offset# channel-stride#)]
                    (calc-squared-sums! input# n-channels# remove-index# add-index#
                                        squared-sum-view# alpha# k# ~cast-fn)
                    (c-for
                     [chan-idx# 0 (< chan-idx# n-channels#) (inc chan-idx#)]
                     (let [divisor# (safe-positive-pow
                                     (.get squared-sum-view# chan-idx#)
                                     beta#)]
                       (.set output# chan-idx# (~cast-fn (/ (.get input# chan-idx#)
                                                            divisor#))))))))
               (catch Throwable e# (clojure.pprint/pprint e#))))]
        ;;(pixel-fn# 0 num-pixels#)
        (launch-parallel-for num-pixels# pixel-fn#)))))


(defmacro cpu-lrn-backward-impl
  "backward pass, calculated using sage:
https://github.com/thinktopic/cortex/blob/local-response-normalization/sage/local-response-normalization.txt"
  [input output-gradient input-gradient
   input-tensor n k alpha beta cast-fn]
  `(let [input# (ArrayView/toView ~input)
         output-gradient# (ArrayView/toView ~output-gradient)
         input-gradient# (ArrayView/toView ~input-gradient)
         input-tensor# ~input-tensor
         n-channels# (long (.channel-count input-tensor#))
         height# (long (.height input-tensor#))
         width# (long (.width input-tensor#))
         n# (~cast-fn (max 1.0 (~cast-fn (min (~cast-fn ~n) n-channels#))))
         k# (~cast-fn ~k)
         alpha# (/ (~cast-fn ~alpha)
                   n#)
         beta# (~cast-fn ~beta)
         neg-beta# (- beta#)
         [batch-size# batch-stride#] (math/batch-shape input-tensor#)
         batch-size# (long batch-size#)
         batch-stride# (long batch-stride#)
         channel-stride# (* width# height#)
         num-pixels# channel-stride#
         look-data# (calculate-look-amounts n#)
         ;;We use a running sum so when going to the next 'n'
         ;;we want to remove the item at 'n - remove-index'
         ;;and add the item at 'n + add-index'.  As such remove-index
         ;;is technically one past the beginning of our range while
         ;;add-index is the last possible valid index of our range.
         remove-index# (long (:remove-index look-data#))
         add-index# (long (:add-index look-data#))]
     (c-for
      [batch-idx# 0 (< batch-idx# batch-size#) (inc batch-idx#)]
      (let [start-offset# (* batch-idx# batch-stride# )
            ;;Create a function that will be used to parallelize the computation
            pixel-fn#
            (fn [^long start# ^long len#]
              (try
               (let [end# (+ start# len#)
                     squared-sum-view# (ArrayView/toView (double-array n-channels#))]
                 (c-for
                  [pix-idx# start# (< pix-idx# end#) (inc pix-idx#)]
                  (let [pix-offset# (+ start-offset# pix-idx#)
                        ;;Setup input to stride over image channels at this pixel.
                        input# (.toStridedView input# pix-offset# channel-stride#)
                        output-gradient# (.toStridedView output-gradient#
                                                         pix-offset# channel-stride#)
                        input-gradient# (.toStridedView input-gradient#
                                                         pix-offset# channel-stride#)]
                    (calc-squared-sums! input# n-channels# remove-index# add-index#
                                       squared-sum-view# alpha# k# ~cast-fn)
                    (c-for
                     [chan-idx# 0 (< chan-idx# n-channels#) (inc chan-idx#)]
                     (let [range-start# (long (max 0 (+ (- chan-idx# remove-index#) 1)))
                           past-range-end# (long (min n-channels# (+ chan-idx# add-index# 1)))]
                       (loop [range-idx# range-start#
                              output-accum# (double 0.0)]
                         (if (< range-idx# past-range-end#)
                           (let [squared-to-beta# (safe-positive-pow
                                                   (.get squared-sum-view#
                                                         range-idx#)
                                                   beta#)
                                 squared-to-beta-minus-one#
                                 (safe-positive-pow
                                  (.get squared-sum-view#
                                        range-idx#)
                                  (- beta# 1.0))


                                 shared-quantity# (/ (* -2.0 squared-to-beta-minus-one#
                                                        (.get input# chan-idx#)
                                                        (.get input# range-idx#)
                                                        alpha# beta#)
                                                     (* squared-to-beta#
                                                        squared-to-beta#))
                                 addend# (double
                                          (if (= range-idx# chan-idx#)
                                            (/ 1.0 squared-to-beta#)
                                            0.0))]
                             (recur (inc range-idx#)
                                    (double
                                     (+ output-accum#
                                        (* (+ shared-quantity# addend#)
                                           (.get output-gradient# range-idx#))))))
                           (.set input-gradient# chan-idx# output-accum#))))))))
               (catch Throwable e# (clojure.pprint/pprint e#))))]
        (launch-parallel-for num-pixels# pixel-fn#)))))


(extend-type DoubleArrayView
  PCPUNetworkImpl
  (cpu-activation-forward [input-buf act-type ^DoubleArrayView output-buf]
    (cpu-act-forward-impl act-type input-buf output-buf double))
  (cpu-activation-backward [input act-type ^DoubleArrayView output
                            ^DoubleArrayView output-gradient
                            ^DoubleArrayView input-gradient]
    (cpu-act-backward-impl act-type input output output-gradient input-gradient double))
  (cpu-softmax-forward [input-buf ^DoubleArrayView output-buf ^long n-input ^long n-channels]
    (cpu-softmax-forward-impl n-input input-buf output-buf n-channels double))
  (cpu-adadelta-step! [gradient ^DoubleArrayView parameters gradient-alpha
                       param-offset decay epsilon ^DoubleArrayView grad-accum
                       ^DoubleArrayView dx-accum]
    (AdadeltaOptimiser/step_d (double gradient-alpha) (.data gradient) (.data parameters)
                              (int param-offset) (double decay) (double epsilon)
                              (.data grad-accum) (.data dx-accum)))
  (cpu-adam-step! [gradient ^DoubleArrayView parameters gradient-alpha param-offset
                   alpha beta1 beta2 epsilon pow-beta1-t pow-beta2-t
                   ^DoubleArrayView m ^DoubleArrayView v]
    (AdamOptimiser/step_d (double gradient-alpha) (.data gradient) (.data parameters)
                          param-offset (double alpha) (double beta1) (double beta2)
                          (double epsilon) (double pow-beta1-t) (double pow-beta2-t)
                          (.data m) (.data v)))
  (cpu-planar-input->convolution! [input ^DoubleArrayView input-convolved
                                   ^ConvLayerConfig conv-config]
    (cpu-planar-input->convolution!-impl input input-convolved conv-config double))
  (cpu-convolution->planar-output! [input-convolved ^DoubleArrayView input-gradient
                                    ^ConvLayerConfig conv-config]
    (cpu-convolution->planar-output!-impl input-convolved input-gradient conv-config double))

  (cpu-fill [buffer value]
    (Arrays/fill (.data buffer) (.offset buffer)
                 (+ (.offset buffer) (.length buffer)) (double value)))
  (cpu-max-pooling-forward [input ^DoubleArrayView output ^ConvLayerConfig conv-config]
    (cpu-max-pooling-forward-impl input output conv-config double))
  (cpu-max-pooling-backward [input ^DoubleArrayView output ^DoubleArrayView input-gradient
                             ^DoubleArrayView output-gradient ^ConvLayerConfig conv-config]
    (cpu-max-pooling-backward-impl input output input-gradient output-gradient conv-config
                                   double))
  (cpu-prepare-bernoulli-dropout [mult-buffer ^FloatArrayView rand-buffer probability]
    (cpu-prepare-bernoulli-impl mult-buffer rand-buffer probability double))
  (cpu-prepare-gaussian-dropout [mult-buffer ^FloatArrayView rand-buffer]
    (cpu-prepare-gaussian-impl mult-buffer rand-buffer double))
  (cpu-bn-calc [^DoubleArrayView input ^DoubleArrayView means ^DoubleArrayView variances
                ^DoubleArrayView scale ^DoubleArrayView bias ^DoubleArrayView output
                batch-size batch-stride]
    (cpu-bn-calc-impl input means variances scale bias output batch-size batch-stride double))
  (cpu-update-means-variances [input
                               ^DoubleArrayView running-means ^DoubleArrayView running-variances
                               ^DoubleArrayView saved-means ^DoubleArrayView saved-variances
                               batch-size batch-stride ave-factor epsilon]
    (cpu-update-means-variances-impl input running-means running-variances
                                     saved-means saved-variances
                                     batch-size batch-stride
                                     ave-factor epsilon double))
  (cpu-bn-backward [input ^DoubleArrayView means ^DoubleArrayView
                    variances ^DoubleArrayView scale
                    ^DoubleArrayView bias ^DoubleArrayView output
                    ^DoubleArrayView scale-gradient
                    ^DoubleArrayView bias-gradient ^DoubleArrayView input-gradient
                    ^DoubleArrayView output-gradient batch-size batch-stride]
    (cpu-bn-backward-impl input means variances scale bias output scale-gradient bias-gradient
                          input-gradient output-gradient batch-size batch-stride double))
  (cpu-lrn-forward [input ^DoubleArrayView output ^Tensor input-tensor n k alpha beta]
    (cpu-lrn-forward-impl input output input-tensor n k alpha beta double))
  (cpu-lrn-backward [input ^DoubleArrayView output-gradient ^DoubleArrayView input-gradient
                     ^Tensor input-tensor n k alpha beta]
    (cpu-lrn-backward-impl input output-gradient input-gradient input-tensor
                           n k alpha beta double)))


(extend-type FloatArrayView
  PCPUNetworkImpl
  (cpu-activation-forward [input-buf act-type ^FloatArrayView output-buf]
    (cpu-act-forward-impl act-type input-buf output-buf float))
  (cpu-activation-backward [input act-type ^FloatArrayView output
                            ^FloatArrayView output-gradient
                            ^FloatArrayView input-gradient]
    (cpu-act-backward-impl act-type input output output-gradient input-gradient float))
  (cpu-softmax-forward [input-buf ^FloatArrayView output-buf ^long n-input ^long n-channels]
    (cpu-softmax-forward-impl n-input input-buf output-buf n-channels float))
  (cpu-adadelta-step! [gradient ^FloatArrayView parameters gradient-alpha
                       param-offset decay epsilon
                       ^FloatArrayView grad-accum ^FloatArrayView dx-accum]
    (AdadeltaOptimiser/step_f (float gradient-alpha) (.data gradient) (.data parameters)
                              (int param-offset) (float decay) (float epsilon)
                              (.data grad-accum) (.data dx-accum)))
  (cpu-adam-step! [gradient ^FloatArrayView parameters gradient-alpha param-offset
                   alpha beta1 beta2 epsilon pow-beta1-t pow-beta2-t
                   ^FloatArrayView m ^FloatArrayView v]
    (AdamOptimiser/step_f (float gradient-alpha) (.data gradient) (.data parameters)
                          param-offset (float alpha) (float beta1) (float beta2) (float epsilon)
                          (float pow-beta1-t) (float pow-beta2-t) (.data m) (.data v)))
  (cpu-planar-input->convolution! [input ^FloatArrayView input-convolved
                                   ^ConvLayerConfig conv-config]
    (cpu-planar-input->convolution!-impl input input-convolved conv-config float))
  (cpu-convolution->planar-output! [input-convolved ^FloatArrayView input-gradient
                                    ^ConvLayerConfig conv-config]
    (cpu-convolution->planar-output!-impl input-convolved input-gradient conv-config float))
  (cpu-fill [buffer value]
    (Arrays/fill (.data buffer) (.offset buffer)
                 (+ (.offset buffer) (.length buffer)) (float value)))
  (cpu-max-pooling-forward [input ^FloatArrayView output ^ConvLayerConfig conv-config]
    (cpu-max-pooling-forward-impl input output conv-config float))
  (cpu-max-pooling-backward [input ^FloatArrayView output ^FloatArrayView input-gradient
                             ^FloatArrayView output-gradient ^ConvLayerConfig conv-config]
    (cpu-max-pooling-backward-impl input output input-gradient output-gradient conv-config
                                   float))
  (cpu-prepare-bernoulli-dropout [mult-buffer ^FloatArrayView rand-buffer probability]
    (cpu-prepare-bernoulli-impl mult-buffer rand-buffer probability float))
  (cpu-prepare-gaussian-dropout [mult-buffer ^FloatArrayView rand-buffer]
    (cpu-prepare-gaussian-impl mult-buffer rand-buffer float))
  (cpu-bn-calc [^FloatArrayView input ^FloatArrayView means ^FloatArrayView variances
                ^FloatArrayView scale ^FloatArrayView bias ^FloatArrayView output
                batch-size batch-stride]
    (cpu-bn-calc-impl input means variances scale bias output batch-size batch-stride float))
  (cpu-update-means-variances [input
                               ^FloatArrayView running-means ^FloatArrayView running-variances
                               ^FloatArrayView saved-means ^FloatArrayView saved-variances
                               batch-size batch-stride ave-factor epsilon]
    (cpu-update-means-variances-impl input running-means running-variances
                                     saved-means saved-variances
                                     batch-size batch-stride
                                     ave-factor epsilon float))
  (cpu-bn-backward [input ^FloatArrayView means ^FloatArrayView variances ^FloatArrayView scale
                    ^FloatArrayView bias ^FloatArrayView output ^FloatArrayView scale-gradient
                    ^FloatArrayView bias-gradient ^FloatArrayView input-gradient
                    ^FloatArrayView output-gradient batch-size batch-stride]
    (cpu-bn-backward-impl input means variances scale bias output scale-gradient bias-gradient
                          input-gradient output-gradient batch-size batch-stride float))
  (cpu-lrn-forward [input ^FloatArrayView output ^Tensor input-tensor n k alpha beta]
    (cpu-lrn-forward-impl input output input-tensor n k alpha beta float))
  (cpu-lrn-backward [input ^FloatArrayView output-gradient ^FloatArrayView input-gradient
                     ^Tensor input-tensor n k alpha beta]
    (cpu-lrn-backward-impl input output-gradient input-gradient input-tensor
                           n k alpha beta float)))


(defrecord ActivationLayer [act-type cpu-stream])


(defn device-array->view
  [dev-ary]
  (dtype/->view (math/device-buffer dev-ary)))


(extend-type ActivationLayer
  nn-backend/PBackendLayer
  (forward! [layer ^DeviceArray input ^DeviceArray output]
    (cpu-drv/with-stream-dispatch (.cpu-stream layer)
      (cpu-activation-forward (device-array->view input) (.act-type layer) (device-array->view output))))
  (backward! [layer ^DeviceArray input ^DeviceArray output
              ^DeviceArray input-gradient ^DeviceArray output-gradient]
    (cpu-drv/with-stream-dispatch (.cpu-stream layer)
      (cpu-activation-backward (device-array->view input) (.act-type layer)
                               (device-array->view output)
                               (device-array->view output-gradient)
                               (device-array->view input-gradient)))))

(defrecord SoftmaxLayer [cpu-stream n-input channels])
(extend-type SoftmaxLayer
  nn-backend/PBackendLayer
  (forward! [layer ^DeviceArray input ^DeviceArray output]
    (cpu-drv/with-stream-dispatch (.cpu-stream layer)
      (cpu-softmax-forward (device-array->view input) (device-array->view output)
                           (.n-input layer) (.channels layer))))
  (backward! [layer ^DeviceArray input ^DeviceArray output
              ^DeviceArray input-gradient ^DeviceArray output-gradient]
    (layers/softmax-backward! (.cpu-stream layer) input-gradient output-gradient)))



(defrecord ConvolutionalLayer [backend ^ConvLayerConfig conv-config
                               input-convolved ones])


(defn create-convolution-matrix
  [^ConvLayerConfig config backend batch-size]
  (let [output-width (conv/get-output-width config :convolutional)
        output-height (conv/get-output-height config :convolutional)
        kernel-stride (* (.k-width config) (.k-height config))
        n-cols (* kernel-stride (.num-in-channels config))
        n-rows (* output-width output-height)]
    (nn-backend/new-array backend [n-rows n-cols] batch-size)))


(defn create-conv-layer [backend ^ConvLayerConfig conv-config ^long batch-size]
  (let [num-out-pixels (* (conv/get-output-width conv-config :convolutional)
                          (conv/get-output-height conv-config :convolutional))]
    (->ConvolutionalLayer backend conv-config
                          (create-convolution-matrix conv-config backend batch-size)
                          (math/allocate-ones (drv/get-driver backend) (drv/get-stream backend)
                                              (dtype/get-datatype backend) num-out-pixels))))


(extend-type ConvolutionalLayer
  nn-backend/PBackendWeightedLayer
  (weighted-forward! [layer input output weights bias]
    (let [^ConvLayerConfig conv-config (.conv-config layer)
          output-width (conv/get-output-width conv-config :convolutional)
          output-height (conv/get-output-height conv-config :convolutional)
          num-out-pixels (* output-width output-height)
          num-out-channels (.num-out-channels conv-config)
          cpu-stream (drv/get-stream (.backend layer))
          current-thread-stream (cpu-drv/create-main-thread-cpu-stream)]
      ;;In parallel process each item of the batch.
      ;;This allows us to take advantage of at least
      ;;*some* multithreading without having to explicity program much of it.
      (cpu-drv/with-stream-dispatch cpu-stream
        (doall (pmap (fn [[input output input-convolved]]
                       (cpu-planar-input->convolution! (device-array->view input)
                                                       (device-array->view input-convolved)
                                                       (.conv-config layer))
                       ;;set the output to the bias...can't think of another way of doing this.
                       (math/gemm current-thread-stream true false
                                  1.0 bias (.ones layer)
                                  0.0 output)
                       (math/gemm current-thread-stream false true
                                  1.0 weights input-convolved
                                  1.0 output))
                     (math/batched-data-to-per-input-data
                      (drv/get-driver (.backend layer))
                      [input output (.input-convolved layer)]))))))

  (weighted-backward! [layer input output weights bias
                       weight-gradient bias-gradient input-gradient output-gradient]
    (let [^ConvLayerConfig conv-config (.conv-config layer)
          output-width (conv/get-output-width conv-config :convolutional)
          output-height (conv/get-output-height conv-config :convolutional)
          num-out-pixels (* output-width output-height)
          num-out-channels (.num-out-channels conv-config)
          io-data (math/batched-data-to-per-input-data (drv/get-driver (.backend layer))
                                                       [input output input-gradient
                                                        output-gradient
                                                        (.input-convolved layer)])
          cpu-stream (drv/get-stream (.backend layer))
          current-thread-stream (cpu-drv/create-main-thread-cpu-stream)]
      (cpu-drv/with-stream-dispatch cpu-stream
        ;;compute the weights gradient.  These aren't too expensive but we cannot easily
        ;;parallelize this because it is an accumulation.
        (doseq [[input output input-gradient output-gradient input-convolved] io-data]
          (let [output-gradient (math/with-tensor output-gradient
                                  (math/create-tensor num-out-channels num-out-pixels))]
           (math/gemm current-thread-stream false true
                      1.0 (.ones layer) output-gradient
                      1.0 bias-gradient)
           (math/gemm current-thread-stream false false
                      1.0 output-gradient input-convolved
                      1.0 weight-gradient)))

        ;;Once weight/bias gradient accumulation is complete...
        ;;In parallel calculate the input gradients and roll/accumulate
        ;;back up into input vectors.  This changes input convolved
        ;;(which was our previously unrolled input) so we have to do
        ;;it *after* the above step.
        (doall (pmap (fn [[input output input-gradient output-gradient input-convolved]]
                       (let [output-gradient (math/with-tensor output-gradient
                                               (math/create-tensor num-out-channels
                                                                   num-out-pixels))]
                         ;;set input gradient at this batch location to empty
                         (cpu-fill (device-array->view input-gradient) 0)
                         (math/gemm current-thread-stream true false
                                    1.0 output-gradient weights
                                    0.0 input-convolved))
                       (cpu-convolution->planar-output! (device-array->view input-convolved)
                                                        (device-array->view input-gradient)
                                                        conv-config))
                     io-data))))))

(defrecord MaxPooling [backend ^ConvLayerConfig conv-config])
(defn create-max-pooling
  [backend ^ConvLayerConfig config batch-size]
  (->MaxPooling backend config))


(extend-type MaxPooling
  nn-backend/PBackendLayer
  (forward! [layer input output]
    (cpu-drv/with-stream-dispatch (drv/get-stream (.backend layer))
      (doall (pmap (fn [[input output]]
                     (cpu-max-pooling-forward (device-array->view input)
                                              (device-array->view output)
                                              (.conv-config layer)))
                   (math/batched-data-to-per-input-data (drv/get-driver (.backend layer))
                                                        [input output])))))
  (backward! [layer input output input-gradient output-gradient]
    (cpu-drv/with-stream-dispatch (drv/get-stream (.backend layer))
      (doall (pmap (fn [[input output input-gradient output-gradient]]
                     (cpu-fill (device-array->view input-gradient) 0)
                     (cpu-max-pooling-backward (device-array->view input)
                                               (device-array->view output)
                                               (device-array->view input-gradient)
                                               (device-array->view output-gradient)
                                               (.conv-config layer)))
                   (math/batched-data-to-per-input-data
                    (drv/get-driver (.backend layer))
                    [input output input-gradient output-gradient]))))))


(defrecord BatchNormalization [backend])
(extend-type BatchNormalization
  nn-backend/PBatchNormalization
  (batch-norm-calc! [layer input running-means running-variances scale bias output epsilon]
    (let [[batch-size batch-stride] (math/batch-shape input)]
     (cpu-drv/with-stream-dispatch (drv/get-stream (:backend layer))
       (cpu-bn-calc (device-array->view input)
                    (device-array->view running-means)
                    (device-array->view running-variances)
                    (device-array->view scale)
                    (device-array->view bias)
                    (device-array->view output)
                    batch-size batch-stride))))
  (batch-norm-forward! [layer input
                        running-means running-variances
                        saved-means saved-variances
                        scale bias output average-factor epsilon]
    (let [[batch-size batch-stride] (math/batch-shape input)]
      (cpu-drv/with-stream-dispatch (drv/get-stream (:backend layer))
       (cpu-update-means-variances (device-array->view input)
                                   (device-array->view running-means)
                                   (device-array->view running-variances)
                                   (device-array->view saved-means)
                                   (device-array->view saved-variances)
                                   batch-size batch-stride
                                   average-factor epsilon)))
    (nn-backend/batch-norm-calc! layer input saved-means saved-variances
                              scale bias output epsilon))
  (batch-norm-backward! [layer input saved-means saved-variances scale bias output
                         scale-gradient bias-gradient input-gradient output-gradient
                         epsilon]
    (let [[batch-size batch-stride] (math/batch-shape input)]
     (cpu-drv/with-stream-dispatch (drv/get-stream (:backend layer))
       (cpu-bn-backward (device-array->view input)
                        (device-array->view saved-means)
                        (device-array->view saved-variances)
                        (device-array->view scale)
                        (device-array->view bias)
                        (device-array->view output)
                        (device-array->view scale-gradient)
                        (device-array->view bias-gradient)
                        (device-array->view input-gradient)
                        (device-array->view output-gradient)
                        batch-size batch-stride)))))


(defrecord LocalResponseNormalization [backend ^long width ^long height ^long n-channels
                                       n k alpha beta]
  nn-backend/PBackendLayer
  (forward! [layer input output]
    (let [^Tensor input-tensor (.tensor ^DeviceArray input)
          data-tensor (math/create-tensor (.batch-size input-tensor)
                                          n-channels
                                          height
                                          width)]
      (cpu-drv/with-stream-dispatch (drv/get-stream backend)
        (cpu-lrn-forward (device-array->view input)
                         (device-array->view output)
                         data-tensor
                         n k alpha beta))))
  (backward! [layer input output input-gradient output-gradient]
    (let [^Tensor input-tensor (.tensor ^DeviceArray input)
          data-tensor (math/create-tensor (.batch-size input-tensor)
                                          n-channels
                                          height
                                          width)]
     (cpu-drv/with-stream-dispatch (drv/get-stream backend)
       (cpu-lrn-backward (device-array->view input)
                         (device-array->view output-gradient)
                         (device-array->view input-gradient)
                         data-tensor
                         n k alpha beta)))))


(defrecord Recurrrent [backend recurrent-type recurrent-directions
                       ^long n-input ^long n-output ^long batch-size
                       multiplied-input
                       multiplied-hidden
                       input-plus-hidden
                       weights-and-biases
                       weight-and-bias-gradients
                       weight-and-bias-keys]
  nn-backend/PRecurrent
  (get-recurrent-weights-and-biases [layer]
    (vec (mapcat weights-and-biases weight-and-bias-keys)))
  (get-recurrent-weight-and-bias-gradients [layer]
    (vec (mapcat weight-and-bias-gradients weight-and-bias-keys)))
  ;;We don't have an external representation of the weights and biases
  ;;so this step is not necessary.
  (copy-implementation-weights-and-biases! [layer weights-and-biases])
  (recurrent-calc! [layer input initial-hidden-state initial-cell-state output]
    ;;First linear translation of the system.
    (comment
     (nn-backend/biased-multiply! backend input (first (:input weights-and-biases))
                               (second (:input weights-and-biases)) multiplied-input)
     (let [output-vec (split-array-into-batches output)
           hidden-vec (split-array-into-batches multiplied-hidden)
           input-vec (split-array-into-batches multiplied-input)
           input-plus-hidden-vec (split-array-into-batches input-plus-hidden)]
       (loop [idx 0
              input-hidden-state initial-hidden-state
              input-cell-state initial-cell-state]
         (let [iter-output (output-vec idx)
               iter-input (input-vec idx)
               iter-temp-out (intermediate-output-vec idx)
               input-plus-hidden (input-plus-hidden-vec idx)]
           (when (< idx batch-size)
             (nn-backend/biased-multiply! backend input-hidden-state
                                       (first (:recurrence weights-and-biases))
                                       (second (:recurrence weights-and-biases))
                                       (hidden-vec idx))
             (math/sum (drv/get-stream backend) 1.0 iter-input 1.0
                       input-plus-hiden)
             (cond
               (contains? #{:relu :tanh} recurrent-type)
               (cpu-drv/with-stream-dispatch (drv/get-stream backend)
                 (cpu-activation-forward (device-array->view iter-temp-out)
                                         recurrent-type
                                         (device-array->view iter-output)))
               :else
               (throw (Exception. "Unrecognized recurrence type")))
             (recur (inc idx) iter-output running-cell-state)))))))
  (recurrent-forward! [layer input hidden-state cell-state output]
    (nn-backend/recurrent-calc! layer input hidden-state cell-state output))
  (recurrent-backward! [layer input hidden-state cell-state
                        hidden-state-gradient cell-state-gradient
                        input-gradient output-gradient]
    (comment
     ;;Chain rule this stuff backwards...
     (cond
       (contains? #{:relu :tanh} recurrent-type)
       (cpu-drv/with-stream-dispatch (drv/get-stream backend)
         (cpu-activation-backward (device-array->view running-hidden-state)
                                  recurrent-type
                                  (device-array->view output)
                                  (device-array->view output-gradient)
                                  (device-array->view temp-out-gradient)))
       :else
       (throw (Exception. "Unrecognized recurrence type")))
     (let [out-gradient-vec (split-array-into-batches temp-out-gradient)
           hidden-state-input (vec (concat [initial-hidden-state-gradient]
                                           (drop-last
                                            (split-array-into-batches output))))
           input-vec (split-array-into-batches input)
           hidden-state-gradients (split-array-into-batches hidden-state-gradient)]
       (c-for
        [idx 0 (< idx batch-size) (inc idx)]
        (let [ridx (- batch-size idx 1)]
          ;;gradient for hidden state
          (nn-backend/biased-multiply-backward! backend (hidden-state-input ridx)
                                             (first (:recurrence weights-and-biases))
                                             (second (:recurrence weights-and-biases))
                                             (out-gradient-vec ridx)
                                             (hidden-state-gradients ridx)
                                             (first (:recurrence weight-and-bias-gradients))
                                             (second (:recurrence weight-and-bias-gradients))
                                             (out-gradient-vec ridx))

          (nn-backend/biased-multiply-backward! backend (input-vec ridx)
                                             (first (:input weights-and-biases))
                                             (second (:input weights-and-biases))
                                             (out-gradient-vec ridx)
                                             (hidden-state-gradients ridx)
                                             (first (:input weight-and-bias-gradients))
                                             (second (:input weight-and-bias-gradients))
                                             (out-gradient-vec ridx))
          )

        )))
    (nn-backend/biased-multiply-backward! backend )
    )
  )




(extend-type CPUBackend
  dtype/PDatatype
  (get-datatype [backend] (.datatype backend))
  drv/PDriverProvider
  (get-driver [backend] (.driver backend))
  drv/PStreamProvider
  (get-stream [backend] (.stream backend))
  nn-backend/PLayerCreation
  (create-layer [backend {:keys [layer-type] :as layer-desc}]
    (cond
      (contains? #{:sigmoid :relu :tanh} layer-type)
      (->ActivationLayer layer-type (.stream backend))
      (= :softmax layer-type)
      (->SoftmaxLayer (.stream backend) (:output-size layer-desc) (get layer-desc :channels 1))
      (= :convolution layer-type)
      (create-conv-layer backend (:conv-config layer-desc) (:batch-size layer-desc))
      (= :max-pooling layer-type)
      (->MaxPooling backend (:conv-config layer-desc))
      (= :batch-normalization layer-type)
      (->BatchNormalization backend)
      (= :local-response-normalization layer-type)
      (->LocalResponseNormalization backend
                                    (long (:width layer-desc))
                                    (long (:height layer-desc))
                                    (long (:n-channels layer-desc))
                                    (:n layer-desc)
                                    (:k layer-desc)
                                    (:alpha layer-desc)
                                    (:beta layer-desc))
      :else (throw (Exception. (str "Unrecognized layer type: " layer-type)))))
  opt/POptimiseBackend
  (adadelta-step! [backend ^DeviceArray gradient ^DeviceArray parameters
                   gradient-alpha param-offset decay epsilon
                   ^DeviceArray grad-sq-accum ^DeviceArray dx-sq-accum]
    (cpu-drv/with-stream-dispatch (.stream backend)
      (cpu-adadelta-step! (device-array->view gradient) (device-array->view parameters)
                          gradient-alpha param-offset decay epsilon
                          (device-array->view grad-sq-accum) (device-array->view dx-sq-accum))))
  (adam-step! [backend ^DeviceArray gradient ^DeviceArray parameters gradient-alpha param-offset
               alpha beta1 beta2 epsilon pow-beta1-t pow-beta2-t ^DeviceArray m ^DeviceArray v]
    (cpu-drv/with-stream-dispatch (.stream backend)
      (cpu-adam-step! (device-array->view gradient) (device-array->view parameters)
                      gradient-alpha param-offset alpha beta1 beta2 epsilon
                      pow-beta1-t pow-beta2-t (device-array->view m) (device-array->view v))))
  nn-backend/PDropout
  (prepare-bernoulli-dropout! [backend probability rand-buffer mult-buffer]
    (cpu-drv/with-stream-dispatch (.stream backend)
      (cpu-prepare-bernoulli-dropout (device-array->view mult-buffer)
                                     (device-array->view rand-buffer) probability)))
  ;;Gaussian distribution copied to mult buffer.
  (prepare-gaussian-dropout! [backend rand-buffer mult-buffer]
    (cpu-drv/with-stream-dispatch (.stream backend)
      (cpu-prepare-gaussian-dropout (device-array->view mult-buffer)
                                    (device-array->view rand-buffer)))))
