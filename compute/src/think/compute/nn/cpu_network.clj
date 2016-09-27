(ns think.compute.nn.cpu-network
  (:require [think.compute.nn.network :as network]
            [think.compute.device :as dev]
            [think.compute.math :as math]
            [think.compute.cpu-device :as cpu-dev]
            [think.compute.datatype :refer [v-aget v-aset v-alength] :as dtype]
            [think.compute.optimise :as opt]
            [think.compute.nn.layers :as layers]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.protocols :as mp]
            [clojure.core.matrix.macros :refer [c-for]]
            [cortex.nn.impl.layers.convolution :as conv])
  (:import [think.compute.cpu_device CPUDevice CPUStream]
           [java.nio DoubleBuffer FloatBuffer]
           [think.compute.math DeviceArray Tensor]
           [think.compute.optimise AdadeltaOptimiser AdamOptimiser]
           [cortex.nn.impl.layers.convolution ConvLayerConfig]
           [java.util Arrays]
           [java.util.concurrent ForkJoinPool Callable Future]
           [think.compute ArrayView]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defrecord CPUNetwork [^CPUDevice device ^CPUStream stream datatype])


(defn create-cpu-network
  (^CPUNetwork [datatype]
   (let [device (cpu-dev/create-device)
         stream (dev/create-stream device)]
     (->CPUNetwork device stream datatype)))
  (^CPUNetwork []
   (create-cpu-network :double)))


(defprotocol PNIONetwork
  "Implementation of various functions based on buffer datatype."
  (nio-activation-forward [input-buf act-type output-buf])
  (nio-activation-backward [input-buf act-type output-buf
                            output-gradient input-gradient])
  (nio-softmax-forward [input-buf output-buf n-input])
  (nio-adadelta-step! [gradient parameters gradient-alpha param-offset
                       decay epsilon grad-accum dx-accum])
  (nio-adam-step! [gradient parameters gradient-alpha param-offset alpha beta1 beta2 epsilon
                   pow_beta1_t pow_beta2_t m v])
  (nio-planar-input->convolution! [input input-convolved conv-config])
  (nio-convolution->planar-output! [input-convolved input-gradient conv-config])
  (nio-fill [buffer value])
  (nio-max-pooling-forward [input output conv-config])
  (nio-max-pooling-backward [input output input-gradient output-gradient conv-config])
  (nio-prepare-bernoulli-dropout [mult-buffer rand-buffer probability])
  (nio-prepare-gaussian-dropout [mult-buffer rand-buffer])
  (nio-bn-calc [input running-means running-variances
                scale bias output batch-size batch-stride])
  (nio-update-means-variances [input
                               running-means running-variances
                               saved-means saved-variances
                               batch-size batch-stride ave-factor epsilon])
  (nio-bn-backward [input saved-means saved-variances scale bias output
                    scale-gradient bias-gradient input-gradient
                    output-gradient batch-size batch-stride]))

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


(defmacro nio-act-forward-impl
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

(defmacro nio-act-backward-impl
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


(defmacro nio-softmax-forward-impl
  [n-input input-buf output-buf cast-fn]
  `(let [~n-input (long ~n-input)
         src# (ArrayView/toView ~input-buf)
         dest# (ArrayView/toView ~output-buf)
         batch-size# (quot (v-alength src#) ~n-input)]
     (c-for [batch-idx# 0 (< batch-idx# batch-size#) (inc batch-idx#)]
            (let [start-offset# (* batch-idx# ~n-input)
                  max-val# (~cast-fn (array-max src# ~n-input start-offset# ~cast-fn))]
              ;;Subtract max for numerical stability
              (c-for [idx# 0 (< idx# ~n-input) (inc idx#)]
                     (v-aset dest# (+ idx# start-offset#)
                             (Math/exp (- (v-aget src# (+ idx# start-offset#)) max-val#))))
              (let [sum-val# (~cast-fn (array-sum dest# ~n-input start-offset#))]
                (c-for [idx# 0 (< idx# ~n-input) (inc idx#)]
                       (v-aset dest# (+ idx# start-offset#)
                               (/ (v-aget dest# (+ idx# start-offset#))
                                  sum-val#))))))))


(defmacro nio-planar-input->convolution!-impl
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


(defmacro nio-convolution->planar-output!-impl
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

(defmacro nio-max-pooling-forward-impl
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


(defmacro nio-max-pooling-backward-impl
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

(defmacro nio-prepare-bernoulli-impl
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


(defmacro nio-prepare-gaussian-impl
  [mult-buffer rand-buffer cast-fn]
  `(let [mult-ary# (ArrayView/toView ~mult-buffer)
         elem-count# (.length mult-ary#)
         rand-ary# (ArrayView/toView ~rand-buffer)]
     (c-for [idx# 0 (< idx# elem-count#) (inc idx#)]
            (v-aset mult-ary# idx#
                    (~cast-fn (v-aget rand-ary# idx#))))))


(defmacro nio-bn-calc-impl
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

(defmacro nio-update-means-variances-impl
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

(defmacro nio-bn-backward-impl [input means variances scale bias output
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
                     (~cast-fn (+ (+ sum-part-1# sum-part-3#) sum-part-2#))))))))))


(extend-type DoubleBuffer
  PNIONetwork
  (nio-activation-forward [input-buf act-type ^DoubleBuffer output-buf]
    (nio-act-forward-impl act-type input-buf output-buf double))
  (nio-activation-backward [input act-type ^DoubleBuffer output
                            ^DoubleBuffer output-gradient
                            ^DoubleBuffer input-gradient]
    (nio-act-backward-impl act-type input output output-gradient input-gradient double))
  (nio-softmax-forward [input-buf ^DoubleBuffer output-buf ^long n-input]
    (nio-softmax-forward-impl n-input input-buf output-buf double))
  (nio-adadelta-step! [gradient ^DoubleBuffer parameters gradient-alpha
                       param-offset decay epsilon ^DoubleBuffer grad-accum
                       ^DoubleBuffer dx-accum]
    (AdadeltaOptimiser/step_d (double gradient-alpha) (.array gradient) (.array parameters)
                              (int param-offset) (double decay) (double epsilon)
                              (.array grad-accum) (.array dx-accum)))
  (nio-adam-step! [gradient ^DoubleBuffer parameters gradient-alpha param-offset
                   alpha beta1 beta2 epsilon pow-beta1-t pow-beta2-t
                   ^DoubleBuffer m ^DoubleBuffer v]
    (AdamOptimiser/step_d (double gradient-alpha) (.array gradient) (.array parameters)
                          param-offset (double alpha) (double beta1) (double beta2)
                          (double epsilon) (double pow-beta1-t) (double pow-beta2-t)
                          (.array m) (.array v)))
  (nio-planar-input->convolution! [input ^DoubleBuffer input-convolved
                                   ^ConvLayerConfig conv-config]
    (nio-planar-input->convolution!-impl input input-convolved conv-config double))
  (nio-convolution->planar-output! [input-convolved ^DoubleBuffer input-gradient
                                    ^ConvLayerConfig conv-config]
    (nio-convolution->planar-output!-impl input-convolved input-gradient conv-config double))

  (nio-fill [buffer value]
    (Arrays/fill (.array buffer) (.arrayOffset buffer)
                 (+ (.arrayOffset buffer) (.remaining buffer)) (double value)))
  (nio-max-pooling-forward [input ^DoubleBuffer output ^ConvLayerConfig conv-config]
    (nio-max-pooling-forward-impl input output conv-config double))
  (nio-max-pooling-backward [input ^DoubleBuffer output ^DoubleBuffer input-gradient
                             ^DoubleBuffer output-gradient ^ConvLayerConfig conv-config]
    (nio-max-pooling-backward-impl input output input-gradient output-gradient conv-config
                                   double))
  (nio-prepare-bernoulli-dropout [mult-buffer ^FloatBuffer rand-buffer probability]
    (nio-prepare-bernoulli-impl mult-buffer rand-buffer probability double))
  (nio-prepare-gaussian-dropout [mult-buffer ^FloatBuffer rand-buffer]
    (nio-prepare-gaussian-impl mult-buffer rand-buffer double))
  (nio-bn-calc [^DoubleBuffer input ^DoubleBuffer means ^DoubleBuffer variances
                ^DoubleBuffer scale ^DoubleBuffer bias ^DoubleBuffer output
                batch-size batch-stride]
    (nio-bn-calc-impl input means variances scale bias output batch-size batch-stride double))
  (nio-update-means-variances [input
                               ^DoubleBuffer running-means ^DoubleBuffer running-variances
                               ^DoubleBuffer saved-means ^DoubleBuffer saved-variances
                               batch-size batch-stride ave-factor epsilon]
    (nio-update-means-variances-impl input running-means running-variances
                                     saved-means saved-variances
                                     batch-size batch-stride
                                     ave-factor epsilon double))
  (nio-bn-backward [input ^DoubleBuffer means ^DoubleBuffer variances ^DoubleBuffer scale
                    ^DoubleBuffer bias ^DoubleBuffer output ^DoubleBuffer scale-gradient
                    ^DoubleBuffer bias-gradient ^DoubleBuffer input-gradient
                    ^DoubleBuffer output-gradient batch-size batch-stride]
    (nio-bn-backward-impl input means variances scale bias output scale-gradient bias-gradient
                          input-gradient output-gradient batch-size batch-stride double)))


(extend-type FloatBuffer
  PNIONetwork
  (nio-activation-forward [input-buf act-type ^FloatBuffer output-buf]
    (nio-act-forward-impl act-type input-buf output-buf float))
  (nio-activation-backward [input act-type ^FloatBuffer output
                            ^FloatBuffer output-gradient
                            ^FloatBuffer input-gradient]
    (nio-act-backward-impl act-type input output output-gradient input-gradient float))
  (nio-softmax-forward [input-buf ^FloatBuffer output-buf ^long n-input]
    (nio-softmax-forward-impl n-input input-buf output-buf float))
  (nio-adadelta-step! [gradient ^FloatBuffer parameters gradient-alpha param-offset decay epsilon
                       ^FloatBuffer grad-accum ^FloatBuffer dx-accum]
    (AdadeltaOptimiser/step_f (float gradient-alpha) (.array gradient) (.array parameters)
                              (int param-offset) (float decay) (float epsilon)
                              (.array grad-accum) (.array dx-accum)))
  (nio-adam-step! [gradient ^FloatBuffer parameters gradient-alpha param-offset
                   alpha beta1 beta2 epsilon pow-beta1-t pow-beta2-t
                   ^FloatBuffer m ^FloatBuffer v]
    (AdamOptimiser/step_f (float gradient-alpha) (.array gradient) (.array parameters)
                          param-offset (float alpha) (float beta1) (float beta2) (float epsilon)
                          (float pow-beta1-t) (float pow-beta2-t) (.array m) (.array v)))
  (nio-planar-input->convolution! [input ^FloatBuffer input-convolved
                                   ^ConvLayerConfig conv-config]
    (nio-planar-input->convolution!-impl input input-convolved conv-config float))
  (nio-convolution->planar-output! [input-convolved ^FloatBuffer input-gradient
                                    ^ConvLayerConfig conv-config]
    (nio-convolution->planar-output!-impl input-convolved input-gradient conv-config float))
  (nio-fill [buffer value]
    (Arrays/fill (.array buffer) (.arrayOffset buffer)
                 (+ (.arrayOffset buffer) (.remaining buffer)) (float value)))
  (nio-max-pooling-forward [input ^FloatBuffer output ^ConvLayerConfig conv-config]
    (nio-max-pooling-forward-impl input output conv-config float))
  (nio-max-pooling-backward [input ^FloatBuffer output ^FloatBuffer input-gradient
                             ^FloatBuffer output-gradient ^ConvLayerConfig conv-config]
    (nio-max-pooling-backward-impl input output input-gradient output-gradient conv-config
                                   float))
  (nio-prepare-bernoulli-dropout [mult-buffer ^FloatBuffer rand-buffer probability]
    (nio-prepare-bernoulli-impl mult-buffer rand-buffer probability float))
  (nio-prepare-gaussian-dropout [mult-buffer ^FloatBuffer rand-buffer]
    (nio-prepare-gaussian-impl mult-buffer rand-buffer float))
  (nio-bn-calc [^FloatBuffer input ^FloatBuffer means ^FloatBuffer variances
                ^FloatBuffer scale ^FloatBuffer bias ^FloatBuffer output
                batch-size batch-stride]
    (nio-bn-calc-impl input means variances scale bias output batch-size batch-stride float))
  (nio-update-means-variances [input
                               ^FloatBuffer running-means ^FloatBuffer running-variances
                               ^FloatBuffer saved-means ^FloatBuffer saved-variances
                               batch-size batch-stride ave-factor epsilon]
    (nio-update-means-variances-impl input running-means running-variances
                                     saved-means saved-variances
                                     batch-size batch-stride
                                     ave-factor epsilon float))
  (nio-bn-backward [input ^FloatBuffer means ^FloatBuffer variances ^FloatBuffer scale
                    ^FloatBuffer bias ^FloatBuffer output ^FloatBuffer scale-gradient
                    ^FloatBuffer bias-gradient ^FloatBuffer input-gradient
                    ^FloatBuffer output-gradient batch-size batch-stride]
    (nio-bn-backward-impl input means variances scale bias output scale-gradient bias-gradient
                          input-gradient output-gradient batch-size batch-stride float)))


(defrecord ActivationLayer [act-type cpu-stream])

(extend-type ActivationLayer
  network/PNetLayer
  (forward! [layer ^DeviceArray input ^DeviceArray output]
    (cpu-dev/with-stream-dispatch (.cpu-stream layer)
      (nio-activation-forward (.device-buffer input) (.act-type layer) (.device-buffer output))))
  (backward! [layer ^DeviceArray input ^DeviceArray output
              ^DeviceArray input-gradient ^DeviceArray output-gradient]
    (cpu-dev/with-stream-dispatch (.cpu-stream layer)
      (nio-activation-backward (.device-buffer input) (.act-type layer)
                               (.device-buffer output)
                               (.device-buffer output-gradient)
                               (.device-buffer input-gradient)))))

(defrecord SoftmaxLayer [cpu-stream n-input])
(extend-type SoftmaxLayer
  network/PNetLayer
  (forward! [layer ^DeviceArray input ^DeviceArray output]
    (cpu-dev/with-stream-dispatch (.cpu-stream layer)
      (nio-softmax-forward (.device-buffer input) (.device-buffer output) (.n-input layer))))
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
    (network/new-array backend [n-rows n-cols] batch-size)))


(defn create-conv-layer [backend ^ConvLayerConfig conv-config ^long batch-size]
  (let [num-out-pixels (* (conv/get-output-width conv-config :convolutional)
                          (conv/get-output-height conv-config :convolutional))]
    (->ConvolutionalLayer backend conv-config
                          (create-convolution-matrix conv-config backend batch-size)
                          (math/allocate-ones (dev/get-device backend) (dev/get-stream backend)
                                              (dtype/get-datatype backend) num-out-pixels))))


(extend-type ConvolutionalLayer
  network/PNetWeightedLayer
  (weighted-forward! [layer input output weights bias]
    (let [^ConvLayerConfig conv-config (.conv-config layer)
          output-width (conv/get-output-width conv-config :convolutional)
          output-height (conv/get-output-height conv-config :convolutional)
          num-out-pixels (* output-width output-height)
          num-out-channels (.num-out-channels conv-config)
          cpu-stream (dev/get-stream (.backend layer))
          current-thread-stream (cpu-dev/create-main-thread-cpu-stream)]
      ;;In parallel process each item of the batch.
      ;;This allows us to take advantage of at least
      ;;*some* multithreading without having to explicity program much of it.
      (cpu-dev/with-stream-dispatch cpu-stream
        (doall (pmap (fn [[input output input-convolved]]
                       (nio-planar-input->convolution! (math/device-buffer input)
                                                       (math/device-buffer input-convolved)
                                                       (.conv-config layer))
                       ;;set the output to the bias...can't think of another way of doing this.
                       (math/gemm current-thread-stream true false
                                  1.0 bias (.ones layer)
                                  0.0 output)
                       (math/gemm current-thread-stream false true
                                  1.0 weights input-convolved
                                  1.0 output))
                     (math/batched-data-to-per-input-data
                      (dev/get-device (.backend layer))
                      [input output (.input-convolved layer)]))))))

  (weighted-backward! [layer input output weights bias
                       weight-gradient bias-gradient input-gradient output-gradient]
    (let [^ConvLayerConfig conv-config (.conv-config layer)
          output-width (conv/get-output-width conv-config :convolutional)
          output-height (conv/get-output-height conv-config :convolutional)
          num-out-pixels (* output-width output-height)
          num-out-channels (.num-out-channels conv-config)
          io-data (math/batched-data-to-per-input-data (dev/get-device (.backend layer))
                                                       [input output input-gradient
                                                        output-gradient
                                                        (.input-convolved layer)])
          cpu-stream (dev/get-stream (.backend layer))
          current-thread-stream (cpu-dev/create-main-thread-cpu-stream)]
      (cpu-dev/with-stream-dispatch cpu-stream
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
                         (nio-fill (math/device-buffer input-gradient) 0)
                         (math/gemm current-thread-stream true false
                                    1.0 output-gradient weights
                                    0.0 input-convolved))
                       (nio-convolution->planar-output! (math/device-buffer input-convolved)
                                                        (math/device-buffer input-gradient)
                                                        conv-config))
                     io-data))))))

(defrecord MaxPooling [backend ^ConvLayerConfig conv-config])
(defn create-max-pooling
  [backend ^ConvLayerConfig config batch-size]
  (->MaxPooling backend config))


(extend-type MaxPooling
  network/PNetLayer
  (forward! [layer input output]
    (cpu-dev/with-stream-dispatch (dev/get-stream (.backend layer))
      (doall (pmap (fn [[input output]]
                     (nio-max-pooling-forward (math/device-buffer input)
                                              (math/device-buffer output)
                                              (.conv-config layer)))
                   (math/batched-data-to-per-input-data (dev/get-device (.backend layer))
                                                        [input output])))))
  (backward! [layer input output input-gradient output-gradient]
    (cpu-dev/with-stream-dispatch (dev/get-stream (.backend layer))
      (doall (pmap (fn [[input output input-gradient output-gradient]]
                     (nio-fill (math/device-buffer input-gradient) 0)
                     (nio-max-pooling-backward (math/device-buffer input)
                                               (math/device-buffer output)
                                               (math/device-buffer input-gradient)
                                               (math/device-buffer output-gradient)
                                               (.conv-config layer)))
                   (math/batched-data-to-per-input-data
                    (dev/get-device (.backend layer))
                    [input output input-gradient output-gradient]))))))


(defrecord BatchNormalization [backend])
(extend-type BatchNormalization
  network/PBatchNormalization
  (batch-norm-calc! [layer input running-means running-variances scale bias output epsilon]
    (let [[batch-size batch-stride] (math/batch-shape input)]
     (cpu-dev/with-stream-dispatch (dev/get-stream (:backend layer))
       (nio-bn-calc (math/device-buffer input)
                    (math/device-buffer running-means)
                    (math/device-buffer running-variances)
                    (math/device-buffer scale)
                    (math/device-buffer bias)
                    (math/device-buffer output)
                    batch-size batch-stride))))
  (batch-norm-forward! [layer input
                        running-means running-variances
                        saved-means saved-variances
                        scale bias output average-factor epsilon]
    (let [[batch-size batch-stride] (math/batch-shape input)]
      (cpu-dev/with-stream-dispatch (dev/get-stream (:backend layer))
       (nio-update-means-variances (math/device-buffer input)
                                   (math/device-buffer running-means)
                                   (math/device-buffer running-variances)
                                   (math/device-buffer saved-means)
                                   (math/device-buffer saved-variances)
                                   batch-size batch-stride
                                   average-factor epsilon)))
    (network/batch-norm-calc! layer input saved-means saved-variances
                              scale bias output epsilon))
  (batch-norm-backward! [layer input saved-means saved-variances scale bias output
                         scale-gradient bias-gradient input-gradient output-gradient
                         epsilon]
    (let [[batch-size batch-stride] (math/batch-shape input)]
     (cpu-dev/with-stream-dispatch (dev/get-stream (:backend layer))
       (nio-bn-backward (math/device-buffer input)
                        (math/device-buffer saved-means)
                        (math/device-buffer saved-variances)
                        (math/device-buffer scale)
                        (math/device-buffer bias)
                        (math/device-buffer output)
                        (math/device-buffer scale-gradient)
                        (math/device-buffer bias-gradient)
                        (math/device-buffer input-gradient)
                        (math/device-buffer output-gradient)
                        batch-size batch-stride)))))


(defrecord Recurrrent [backend recurrent-type recurrent-directions
                       ^long n-input ^long n-output ^long batch-size
                       multiplied-input
                       multiplied-hidden
                       input-plus-hidden
                       weights-and-biases
                       weight-and-bias-gradients
                       weight-and-bias-keys]
  network/PRecurrent
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
     (network/biased-multiply! backend input (first (:input weights-and-biases))
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
             (network/biased-multiply! backend input-hidden-state
                                       (first (:recurrence weights-and-biases))
                                       (second (:recurrence weights-and-biases))
                                       (hidden-vec idx))
             (math/sum (dev/get-stream backend) 1.0 iter-input 1.0
                       input-plus-hiden)
             (cond
               (contains? #{:relu :tanh} recurrent-type)
               (cpu-dev/with-stream-dispatch (dev/get-stream backend)
                 (nio-activation-forward (math/device-buffer iter-temp-out)
                                         recurrent-type
                                         (math/device-buffer iter-output)))
               :else
               (throw (Exception. "Unrecognized recurrence type")))
             (recur (inc idx) iter-output running-cell-state)))))))
  (recurrent-forward! [layer input hidden-state cell-state output]
    (network/recurrent-calc! layer input hidden-state cell-state output))
  (recurrent-backward! [layer input hidden-state cell-state
                        hidden-state-gradient cell-state-gradient
                        input-gradient output-gradient]
    (comment
     ;;Chain rule this stuff backwards...
     (cond
       (contains? #{:relu :tanh} recurrent-type)
       (cpu-dev/with-stream-dispatch (dev/get-stream backend)
         (nio-activation-backward (math/device-buffer running-hidden-state)
                                  recurrent-type
                                  (math/device-buffer output)
                                  (math/device-buffer output-gradient)
                                  (math/device-buffer temp-out-gradient)))
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
          (network/biased-multiply-backward! backend (hidden-state-input ridx)
                                             (first (:recurrence weights-and-biases))
                                             (second (:recurrence weights-and-biases))
                                             (out-gradient-vec ridx)
                                             (hidden-state-gradients ridx)
                                             (first (:recurrence weight-and-bias-gradients))
                                             (second (:recurrence weight-and-bias-gradients))
                                             (out-gradient-vec ridx))

          (network/biased-multiply-backward! backend (input-vec ridx)
                                             (first (:input weights-and-biases))
                                             (second (:input weights-and-biases))
                                             (out-gradient-vec ridx)
                                             (hidden-state-gradients ridx)
                                             (first (:input weight-and-bias-gradients))
                                             (second (:input weight-and-bias-gradients))
                                             (out-gradient-vec ridx))
          )

        )))
    (network/biased-multiply-backward! backend )
    )
  )




(extend-type CPUNetwork
  dtype/PDatatype
  (get-datatype [net] (.datatype net))
  dev/PDeviceProvider
  (get-device [net] (.device net))
  dev/PStreamProvider
  (get-stream [net] (.stream net))
  network/PLayerCreation
  (create-layer [net {:keys [layer-type] :as layer-desc}]
    (cond
      (contains? #{:sigmoid :relu :tanh} layer-type)
      (->ActivationLayer layer-type (.stream net))
      (= :softmax layer-type)
      (->SoftmaxLayer (.stream net) (:output-size layer-desc))
      (= :convolution layer-type)
      (create-conv-layer net (:conv-config layer-desc) (:batch-size layer-desc))
      (= :max-pooling layer-type)
      (->MaxPooling net (:conv-config layer-desc))
      (= :batch-normalization layer-type)
      (->BatchNormalization net)
      :else (throw (Exception. (str "Unrecognized layer type: " layer-type)))))
  opt/POptimiseBackend
  (adadelta-step! [backend ^DeviceArray gradient ^DeviceArray parameters
                   gradient-alpha param-offset decay epsilon
                   ^DeviceArray grad-sq-accum ^DeviceArray dx-sq-accum]
    (cpu-dev/with-stream-dispatch (.stream backend)
      (nio-adadelta-step! (.device-buffer gradient) (.device-buffer parameters)
                          gradient-alpha param-offset decay epsilon
                          (.device-buffer grad-sq-accum) (.device-buffer dx-sq-accum))))
  (adam-step! [backend ^DeviceArray gradient ^DeviceArray parameters gradient-alpha param-offset
               alpha beta1 beta2 epsilon pow-beta1-t pow-beta2-t ^DeviceArray m ^DeviceArray v]
    (cpu-dev/with-stream-dispatch (.stream backend)
      (nio-adam-step! (.device-buffer gradient) (.device-buffer parameters)
                      gradient-alpha param-offset alpha beta1 beta2 epsilon
                      pow-beta1-t pow-beta2-t (.device-buffer m) (.device-buffer v))))
  network/PDropout
  (prepare-bernoulli-dropout! [backend probability rand-buffer mult-buffer]
    (cpu-dev/with-stream-dispatch (.stream backend)
      (nio-prepare-bernoulli-dropout (math/device-buffer mult-buffer)
                                     (math/device-buffer rand-buffer) probability)))
  ;;Gaussian distribution copied to mult buffer.
  (prepare-gaussian-dropout! [backend rand-buffer mult-buffer]
    (cpu-dev/with-stream-dispatch (.stream backend)
      (nio-prepare-gaussian-dropout (math/device-buffer mult-buffer)
                                    (math/device-buffer rand-buffer)))))
