(ns cortex.impl.layers.convolution
  (:require [cortex.protocols :as cp]
            [cortex.util :as util :refer [error EMPTY-VECTOR]]
            [clojure.core.matrix :as m]
            [core.blas.protocols :as blas]
            [clojure.core.matrix.protocols :as mp]
            [cortex.backends :as b]
            [cortex.impl.layers :as layers]
            #?(:clj [cortex.impl.vectorz-blas])
            #?(:clj [cortex.registry :refer [register-module]]
               :cljs [cortex.registry :refer-macros [register-module]])
            #?(:clj [clojure.core.matrix.macros :refer [c-for]]
               :cljs [clojure.core.matrix.macros :refer-macros [c-for]]))
  #?(:clj (:import [java.util PriorityQueue]
                   [cortex.impl ConvOps])))


(defn interleaved->planar!
  "in-place mutation to take channel data and convert to planar"
  [input output width height num-channels]
  (let [width (long width)
        height (long height)
        num-channels (long num-channels)
        ^doubles in-ary (mp/as-double-array input)
        ^doubles out-ary (mp/as-double-array output)]
    (if (= 1 num-channels)
      (m/assign! output input)
      (let [plane-stride (* width height)
            input-stride (* width num-channels)]
        (c-for
         [h 0 (< h height) (inc h)]
         (c-for
          [w 0 (< w width) (inc w)]
          (c-for
           [chan 0 (< chan num-channels) (inc chan)]
           (let [input-pixel (+ (* h input-stride)
                                (* w num-channels))
                 output-pixel (+ (* h width) w
                                 (* chan plane-stride))]
             (aset out-ary output-pixel (aget in-ary input-pixel))))))))
    output))


(defn get-padded-strided-dimension
  "http://caffe.berkeleyvision.org/tutorial/layers.html.  Returns the dimensions
of the output of a conv-net ignoring channels."
  ^long [^long input-dim ^long pad ^long kernel-size ^long stride]
  (long (+ (quot (- (+ input-dim (* 2 pad))  kernel-size)
                 stride)
           1)))


(defrecord ConvLayerConfig
    [^long width ^long height ^long k-width ^long k-height
     ^long padx ^long pady ^long stride-w ^long stride-h
     ^long num-in-channels ^long num-out-channels])


(defn get-output-width
  ^long [^ConvLayerConfig conv]
  (get-padded-strided-dimension (.width conv) (.padx conv) (.k-width conv) (.stride-w conv)))

(defn get-output-height
  ^long [^ConvLayerConfig conv]
  (get-padded-strided-dimension (.height conv) (.pady conv) (.k-height conv) (.stride-h conv)))


(defn create-conv-layer-config
  ([width height kernel-width kernel-height
    padx pady stride-w stride-h num-in-channels
    num-out-channels]
   (->ConvLayerConfig width height kernel-width kernel-height
                      padx pady stride-w stride-h
                      num-in-channels num-out-channels))
  ([width height kernel-width kernel-height
    padx pady stride-w stride-h num-in-channels]
   (create-conv-layer-config width height kernel-width kernel-height
                             padx pady stride-w stride-h num-in-channels
                             num-in-channels)))

(defn create-convolution-matrix
  [^ConvLayerConfig config]
  (let [output-width (get-output-width config)
        output-height (get-output-height config)
        kernel-stride (* (.k-width config) (.k-height config))
        n-cols (* kernel-stride (.num-in-channels config))
        n-rows (* output-width output-height)]
    (b/new-array [n-rows n-cols])))


(defmacro convolution-outer-kernel
  [config & body]
  `(let [^ConvLayerConfig config# ~config
         ~'output-width (get-output-width config#)
         ~'output-height (get-output-height config#)
         ~'num-in-channels (.num-in-channels config#)
         ~'num-out-channels (.num-out-channels config#)
         ~'input-planar-stride (* (.width config#) (.height config#))
         ~'output-planar-stride (* ~'output-width ~'output-height ~'num-out-channels)
         ~'output-channel-stride (* (.k-width config#) (.k-height config#))
         ~'output-column-stride (* ~'output-channel-stride ~'num-in-channels)
         ~'width (.width config#)
         ~'height (.height config#)
         ~'k-width (.k-width config#)
         ~'k-height (.k-height config#)
         ~'stride-h (.stride-h config#)
         ~'stride-w (.stride-w config#)
         ~'padx (.padx config#)
         ~'pady (.pady config#)]
     (c-for
      [~'chan 0 (< ~'chan ~'num-in-channels) (inc ~'chan)]
      (let [~'chan-input-offset (* ~'chan ~'input-planar-stride)
            ~'chan-output-offset (* ~'chan ~'output-planar-stride)]
       (c-for
        [~'out-y 0 (< ~'out-y ~'output-height) (inc ~'out-y)]
        (let [~'input-rel-y (- (* ~'out-y ~'stride-h) ~'pady)]
          (c-for
           [~'out-x 0 (< ~'out-x ~'output-width) (inc ~'out-x)]
           (let [~'input-rel-x (- (* ~'out-x ~'stride-w) ~'padx)]
             ~@body))))))))


(defmacro convolution-roll-unroll-inner-kernel
  [& body]
  `(let [~'chan-conv-offset (* ~'chan ~'output-channel-stride)
         ~'output-offset (+ (* ~'out-y ~'output-width)
                            ~'out-x)]
    (c-for
     [~'k-y 0 (< ~'k-y ~'k-height) (inc ~'k-y)]
     (c-for
      [~'k-x 0 (< ~'k-x ~'k-width) (inc ~'k-x)]
      (let [~'input-x (+ ~'input-rel-x ~'k-x)
            ~'input-y (+ ~'input-rel-y ~'k-y)
            ~'output-conv-addr (+ (* ~'output-offset
                                     ~'output-column-stride)
                                  ~'chan-conv-offset
                                  (* ~'k-y ~'k-width)
                                  ~'k-x)
            ~'input-addr  (+ (* ~'input-y ~'width)
                             ~'input-x
                             ~'chan-input-offset)
            ~'input-valid? (and (>= ~'input-x 0)
                                (< ~'input-x ~'width)
                                (>= ~'input-y 0)
                                (< ~'input-y ~'height))]
        ~@body)))))



(defn planar-input->convolution!
  [input output ^ConvLayerConfig config]
  (let [^doubles input-ary (mp/to-double-array input)
        ^doubles output-ary (mp/as-double-array output)]
    (convolution-outer-kernel
     config
     (convolution-roll-unroll-inner-kernel
      (let [input-val (double (if input-valid?
                                (aget input-ary input-addr)
                                0.0))]
        (aset output-ary output-conv-addr input-val)))))
  output)

(defn planar-input->convolution
  [input ^ConvLayerConfig config]
  (let [output (create-convolution-matrix config)]
    (planar-input->convolution! input output config)))


(defn convolution->planar-output!
  "Sum the convolution up to the planar input."
  [conv-input-gradient input-gradient ^ConvLayerConfig config]
  ;;I am using input to mean upstream or in this case destination so that
  ;;this code can look as similar to the code above as possible
  (let [^doubles input-ary (mp/as-double-array input-gradient)
        ^doubles output-ary (mp/as-double-array conv-input-gradient)]
    ;;Zero accumulator
    (m/fill! input-gradient 0.0)
    (convolution-outer-kernel
     config
     (convolution-roll-unroll-inner-kernel
      (when input-valid?
        (let [input-val (aget input-ary input-addr)
              output-val (aget output-ary output-conv-addr)]
          (aset input-ary input-addr (+ input-val output-val)))))))
  input-gradient)



(defn planar-convolution-forward!
  [layer input]
  (let [weights (:weights layer)
        bias (:bias layer)
        ^ConvLayerConfig config (:conv-config layer)
        conv-matrix (or (:conv-matrix layer)
                        (create-convolution-matrix config))
        output-width (get-output-width config)
        output-height (get-output-height config)
        output-channel-stride (* output-width output-height)
        output (or (:output layer)
                   (b/new-array [(.num-out-channels config) output-channel-stride]))
        ones (or (:ones layer)
                 (m/fill! (b/new-array [1 output-channel-stride])
                          1.0))]
    (planar-input->convolution! input conv-matrix config)
    (if (blas/supports-blas? weights)
      (do
        (blas/gemm! output true false 1.0 bias ones 0.0)
        (blas/gemm! output false true 1.0 weights conv-matrix 1.0))
      (do
        (m/assign! output (m/inner-product weights (m/transpose conv-matrix)))
        (m/add! output (m/inner-product (m/transpose bias) ones))))
    (assoc layer
           :conv-matrix conv-matrix
           :output output
           :ones ones)))


(defn planar-convolution-backward!
  [layer input output-gradient]
  (let [weights (:weights layer)
        bias (:bias layer)
        ^ConvLayerConfig config (:conv-config layer)
        output-width (get-output-width config)
        output-height (get-output-height config)
        n-output (* output-width output-height)
        n-kernels (.num-out-channels config)
        conv-matrix (:conv-matrix layer)
        ones (:ones layer)
        output-channel-stride (* output-width output-height)
        weight-gradient (or (:weight-gradient layer)
                            (b/new-array (m/shape weights)))
        bias-gradient (or (:bias-gradient layer)
                          (b/new-array (m/shape bias)))
        output-gradient-matrix (or (:output-gradient-matrix layer)
                                   (b/new-array [n-kernels n-output]))
        input-gradient (or (:input-gradient layer)
                           (b/new-array (m/shape input)))]
    (m/assign! (m/as-vector output-gradient-matrix) output-gradient)
    ;;conv-matrix is assumed to hold the actual input
    (if (blas/supports-blas? weights)
      (do
        (blas/gemm! bias-gradient false true 1.0 ones output-gradient-matrix 1.0)
        (blas/gemm! weight-gradient false false 1.0 output-gradient-matrix conv-matrix 1.0)
        (blas/gemm! conv-matrix true false 1.0 output-gradient-matrix weights 0.0))
      (do
        (let [output-gradient-transpose (m/transpose output-gradient-matrix)]
          (m/add! bias-gradient (m/inner-product ones output-gradient-transpose))
          (m/add! weight-gradient (m/inner-product output-gradient-matrix conv-matrix))
          (m/assign! conv-matrix (m/inner-product output-gradient-transpose weights)))))
    (convolution->planar-output! conv-matrix input-gradient config)
    (assoc layer
           :weight-gradient weight-gradient
           :bias-gradient bias-gradient
           :output-gradient-matrix output-gradient-matrix
           :input-gradient input-gradient)))


;; Forward: Take the input which is expected to be a single vector of data
;; and create a matrix that contains a row for each convolution.  So for example if you have
;; 2x2 kernels and you have a 3x3 matrix of monotonically incrementing indexes we produce
;; a new matrix
;; input: [[1 2 3]
;;         [4 5 6]
;;         [7 8 9]]
;;
;;
;; convolved rows:
;;
;; [[1.0,2.0,4.0,5.0],
;;  [2.0,3.0,5.0,6.0],
;;  [4.0,5.0,7.0,8.0],
;;  [5.0,6.0,8.0,9.0]].
;;
;; You can see how each convolution is represented by a row in the matrix.
;;
;; Now we expect our weights and bias in the same format as what the linear layer
;; uses so each convolution kernel has a row of weights and a bias entry in the bias
;; vector.
;;


#?(:cljs (register-module cortex.impl.layers.Convolutional))
(defrecord Convolutional [weights bias conv-config]
    cp/PModule
    (cp/calc [this input]
      (planar-convolution-forward! this input))

    (cp/output [m]
      (m/as-vector (:output m)))

    cp/PNeuralTraining
    (forward [this input]
      (cp/calc this input))

    (backward [this input output-gradient]
      (planar-convolution-backward! this input output-gradient))

    (input-gradient [this]
      (m/as-vector (:input-gradient this)))

    cp/PParameters
    (parameters [this]
      (m/join (m/as-vector (:weights this)) (m/as-vector (:bias this))))

    (update-parameters [this parameters]
      (let [param-view (cp/parameters this)]
        (m/assign! param-view parameters))
      (let [gradient-view (cp/gradient this)]
        (m/assign! gradient-view 0.0))
      this)

    cp/PGradient
    (gradient [this]
      (m/join (m/as-vector (:weight-gradient this)) (m/as-vector (:bias-gradient this)))))


(defn planar-max-pooling-forward!
  [layer input]
  (let [^ConvLayerConfig config (:conv-config layer)
        output-width (get-output-width config)
        output-height (get-output-height config)
        n-output (* output-width output-height (.num-in-channels config))
        output (or (:output layer)
                   (b/new-array [n-output]))
        ^ints output-indexes (or (:output-indexes layer)
                                 (int-array n-output))
        ^doubles input-ary (mp/as-double-array input)
        ^doubles output-ary (mp/as-double-array output)]
    (convolution-outer-kernel
     config
     (convolution-roll-unroll-inner-kernel
      (let [input-val (double (if input-valid?
                                (aget input-ary input-addr)
                                0.0))
            output-addr (+ (* out-y output-width)
                           out-x
                           chan-output-offset)
            k-idx (+ (* k-y k-width) k-x)
            output-val (aget output-ary output-addr)]
        (when (or (= 0 k-idx)
                  (> input-val output-val))
          (aset output-indexes output-addr k-idx)
          (aset output-ary output-addr output-val)))))
    (assoc layer
           :output output
           :output-indexes output-indexes)))


(defn planar-max-pooling-backward!
  "Calculates the input gradient using the inverse of the convolution step
  combined with the output gradient and the output indexes which tell you which kernel
  index the output came from."
  [layer input output-gradient]
  (let [^ConvLayerConfig config (:conv-config layer)
        ^ints output-indexes (:output-indexes layer)
        input-gradient (or (:input-gradient layer)
                           (b/new-array (m/shape input)))
        ^doubles input-ary (mp/as-double-array input-gradient)
        ^doubles output-ary (mp/as-double-array output-gradient)]
    ;;Zero accumulator
    (m/fill! input-gradient 0.0)
    (convolution-outer-kernel
     config
     (let [output-addr (+ (* out-y output-width)
                          out-x
                          chan-output-offset)
           k-idx (aget output-indexes output-addr)
           output-val (aget output-ary output-addr)
           k-y (quot k-idx k-width)
           k-x (rem k-idx k-width)
           input-x (+ input-rel-x k-x)
           input-y (+ input-rel-y k-y)
           addr (+ (* input-y width)
                   input-x
                   chan-input-offset)]
       (when (and (> input-x 0)
                  (<= input-x width)
                  (> input-y 0)
                  (<= input-y height))
         (aset input-ary addr output-val))))
    (assoc layer :input-gradient input-gradient)))

;;Max pooling layer.  There are other pooling layer types (average,a sochiastic)
;;that may be implemented later but for now we only need max pooling.
#?(:cljs (register-module cortex.impl.layers.Pooling))
(defrecord Pooling [conv-config]
  cp/PModule
  (cp/calc [this input]
    (planar-max-pooling-forward! this input))

  (cp/output [m]
    (:output m))

  cp/PNeuralTraining
  (forward [this input]
    (cp/calc this input))

  (backward [this input output-gradient]
    (planar-max-pooling-backward! this input output-gradient))

  (input-gradient [this]
    (:input-gradient this)))
