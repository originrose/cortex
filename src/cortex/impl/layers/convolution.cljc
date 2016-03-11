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
     ^long num-kernels ^long num-channels])


(defn output-width
  ^long [^ConvLayerConfig conv]
  (get-padded-strided-dimension (.width conv) (.padx conv) (.k-width conv) (.stride-w conv)))

(defn output-height
  ^long [^ConvLayerConfig conv]
  (get-padded-strided-dimension (.height conv) (.pady conv) (.k-height conv) (.stride-h conv)))


(defn create-conv-layer-config
  [width height kernel-width kernel-height
   padx pady stride-w stride-h num-channels
   num-kernels]
  (->ConvLayerConfig width height kernel-width kernel-height
                     padx pady stride-w stride-h num-kernels
                     num-channels))



(defn planar-input->convolution
  [input output ^ConvLayerConfig config]
  (let [^doubles input-ary (mp/as-double-array input)
        ^doubles output-ary (mp/as-double-array output)
        output-width (output-width config)
        output-height (output-height config)
        num-channels (.num-channels config)
        input-planar-stride (* (.width config) (.height config))
        output-channel-stride (* (.k-width config) (.k-height config))
        output-column-stride (* output-channel-stride num-channels)
        width (.width config)
        height (.height config)
        k-width (.k-width config)
        k-height (.k-height config)
        stride-h (.stride-h config)
        stride-w (.stride-w config)
        padx (.padx config)
        pady (.pady config)]
    (c-for
     [out-h 0 (< out-h output-height) (inc out-h)]
     (let [input-rel-y (- (* out-h stride-h) pady)]
      (c-for
       [out-w 0 (< out-w output-width) (inc out-w)]
       (let [input-rel-x (- (* out-w stride-w) padx)]
        (c-for
         [chan 0 (< chan num-channels) (inc chan)]
         (let [chan-input-offset (* chan input-planar-stride)
               chan-output-offset (* chan output-channel-stride)]
          (c-for
           [k-y 0 (< k-y k-height) (inc k-y)]
           (c-for
            [k-x 0 (< k-x k-width) (inc k-x)]
            (let [input-x (+ input-rel-x k-x)
                  input-y (+ input-rel-y k-y)
                  input-val (double (if (and (> input-x 0)
                                             (<= input-x width)
                                             (> input-y 0)
                                             (<= input-y height))
                                      (let [addr (+ (* input-y width)
                                                    input-x
                                                    chan-input-offset)]
                                        (aget input-ary addr)
                                        0.0)))
                  output-addr (+ (* out-h out-w output-column-stride)
                                 chan-output-offset
                                 (* k-y k-width)
                                 k-x)]
              (aset output-ary output-addr input-val)))))))))))
  output)

(defn columnar-sum
  "Sum the columns of the matrix so we produce in essence a new row
which contains the columnar sums.  Produces a vector and leaves input unchanged"
  [m]
  (let [rows (m/rows m)
        accumulator (m/clone (first rows))]
    (reduce m/add! accumulator (rest rows))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Convolution layer forward pass utility functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn create-padded-matrix
  "We need to force vectorz to create a dense matrix.  Hence the superfluous lines"
  [{:keys [^long width ^long height ^long padx ^long pady
           ^long num-channels] :as conv-layer-config}]
  (m/array :vectorz (repeat (+ height (* 2 pady))
                            (repeat (* (+ width (* 2 padx))
                                       num-channels)
                                    0.0))))


(defn sub-vector-assignment!
  [padx pady image-width num-channels backing-matrix input-vector]
  (let [^long padx padx
        ^long pady pady
        ^long image-width image-width
        ^long num-channels num-channels
        input-row-stride (* image-width num-channels)
        input-row-count (quot (long (first (m/shape input-vector))) input-row-stride)]
    (loop [row 0]
      (when (< row input-row-count)
        (let [backing-row-idx (+ pady row)
              backing-row (m/get-row backing-matrix backing-row-idx)
              input-row (m/subvector input-vector (* row input-row-stride) input-row-stride)
              backing-row (m/subvector backing-row (* padx num-channels) input-row-stride)]
          (m/assign! backing-row input-row))
        (recur (inc row))))))


(defn create-padded-input-matrix
  "Remove padding from the equation by creating an input vector that includes it
and then copying the input data (if necessary) into the input vector.  Note this operation
is done for every input *and* it is done for the max pooling layers.  If you want to avoid
the perf hit of a copy then don't use padding."
  [interleaved-input-vector {:keys [^long width ^long height ^long padx ^long pady
                                    ^long num-channels] :as conv-layer-config}]
  (let [has-padding (or (> padx 0) (> pady 0))]
    (if has-padding
      (let [padded-input-matrix (create-padded-matrix conv-layer-config)]
        (sub-vector-assignment! padx pady width num-channels padded-input-matrix interleaved-input-vector)
        padded-input-matrix)
      (m/reshape interleaved-input-vector [height (* width num-channels)]))))


(def convolution-operation-sequence
  (memoize
   (fn [{:keys [^long width ^long height
                ^long k-width ^long k-height
                ^long padx ^long pady
                ^long stride-w ^long stride-h
                ^long num-channels] :as conv-layer-config}]
     (let [output-width (get-padded-strided-dimension width padx k-width stride-w)
           output-height (get-padded-strided-dimension height pady k-height stride-w)
           ;;A 'row' for each output pixel
           num-rows (* output-width output-height)]
       (flatten
        (for [^long idx (range num-rows)]
          (let [output-x (rem idx output-width)
                output-y (quot idx output-width)
                input-left (- (* output-x stride-w) padx)
                input-top (- (* output-y stride-h) pady)]
            (for [^long conv-y (range k-height)]
              (let [input-y (+ input-top conv-y)
                    input-x input-left]
                {:input-x input-x :input-y input-y :conv-x 0 :conv-y conv-y
                 :output-x output-x :output-y output-y})))))))))


(def convolution-input-sequence
  (memoize
   (fn [{:keys [^long width ^long height
                ^long k-width ^long k-height
                ^long padx ^long pady
                ^long stride-w ^long stride-h
                ^long num-channels] :as conv-layer-config}]
     ;;Trim out rows that are completely invalid w/r/t the input non-padded matrix
     (let [valid-rows (filter #(let [{:keys [^long input-y ^long input-x]} %]
                                 (and (>= input-y 0)
                                      (< input-y height)
                                      (>= (+ input-x k-width) 0)
                                      (< input-x width)))
                              (convolution-operation-sequence conv-layer-config))
           output-width (get-padded-strided-dimension width padx k-width stride-w)
           output-height (get-padded-strided-dimension height pady k-height stride-w)]
       ;;Transform each entry such that the x-op stays in bounds.  Also, change output-x,output-y
       ;;into an output-row-index that indicates which row in the convolution matrix this item corresponds to.
       (map (fn [{:keys [^long input-x ^long input-y ^long conv-x ^long conv-y ^long output-x ^long output-y]
                  :as conv-window}]
              (let [input-offset-x (max input-x 0)
                    conv-x (- input-offset-x input-x)
                    k-width (- k-width conv-x)
                    input-end (min width (+ input-offset-x k-width))
                    write-len (- input-end input-offset-x)
                    conv-row-index (+ (* output-y output-width) output-x)]
                (assoc conv-window :conv-x conv-x :input-x input-offset-x
                       :write-len write-len :conv-row-index conv-row-index)))
            valid-rows)))))


(defn convolution-to-input-vector!
  [conv-matrix input-vector {:keys [^long width ^long height
                                    ^long k-width ^long k-height
                                    ^long padx ^long pady
                                    ^long stride-w ^long stride-h
                                    ^long num-channels] :as conv-layer-config}]
  (let [kernel-stride (* k-width num-channels)
        input-stride (* width num-channels)]
    (doseq [input-item (convolution-input-sequence conv-layer-config)]
      (let [{:keys [^long input-x ^long input-y
                    ^long conv-x ^long conv-y
                    ^long write-len ^long conv-row-index]} input-item
            input-row (m/get-row conv-matrix conv-row-index)
            write-idx 0
            conv-offset (+ (* conv-y kernel-stride)
                           (* (+ conv-x write-idx) num-channels))
            input-offset (+ (* input-y input-stride)
                            (* (+ input-x write-idx) num-channels))
            write-num-elems (* write-len num-channels)]
        (m/add! (m/subvector input-vector input-offset write-num-elems)
                (m/subvector input-row conv-offset write-num-elems))))
    input-vector))


(defn input-vector-to-convolution!
  [input-vector conv-matrix {:keys [^long width ^long height
                                    ^long k-width ^long k-height
                                    ^long padx ^long pady
                                    ^long stride-w ^long stride-h
                                    ^long num-channels] :as conv-layer-config}]
  (let [kernel-stride (* k-width num-channels)
        input-stride (* width num-channels)
        num-rows (m/row-count conv-matrix)]
    (doseq [input-item (convolution-input-sequence conv-layer-config)]
      (let [{:keys [^long input-x ^long input-y
                    ^long conv-x ^long conv-y
                    ^long write-len ^long conv-row-index]} input-item
            conv-row (m/get-row conv-matrix conv-row-index)
            write-idx 0
            conv-offset (+ (* conv-y kernel-stride)
                           (* (+ conv-x write-idx) num-channels))
            input-offset (+ (* input-y input-stride)
                            (* (+ input-x write-idx) num-channels))
            write-num-elems (* write-len num-channels)]
        (m/assign! (m/subvector conv-row conv-offset write-num-elems)
                   (m/subvector input-vector input-offset write-num-elems))))
    conv-matrix))


(defn convolution-sequence
  "Produce a sequence of views that performs a convolution over an input matrix"
  [input-matrix output-width output-height {:keys [^long k-width ^long k-height
                                                   ^long stride-w ^long stride-h
                                                   ^long num-channels]
                                            :as conv-layer-config}]
  (let [kernel-stride (* k-width num-channels)]
    (for [^long output-y (range output-height)
          ^long output-x (range output-width)]
      (m/as-vector (m/submatrix input-matrix [[(* output-y stride-h) k-height]
                                              [(* output-x stride-w num-channels) kernel-stride]])))))


(defn create-convolution-rows
  "Given an image flattened into an interleaved vector
create a sequence where each item is the input to the convolution filter row
meaning the convolution is just a dotproduct across the rows.
Should be output-width*output-height rows.  Padding is applied as zeros across channels."
  [interleaved-input-vector {:keys [^long width ^long height ^long k-width ^long k-height
                                    ^long padx ^long pady ^long stride-w ^long stride-h
                                    ^long num-channels] :as conv-layer-config }]
  (let [input-matrix (create-padded-input-matrix interleaved-input-vector
                                                 conv-layer-config)
        output-width (get-padded-strided-dimension width padx k-width stride-w)
        output-height (get-padded-strided-dimension height pady k-height stride-h)
        input-mat-stride (* (+ width (* 2 padx)) num-channels)
        kernel-stride (* k-width num-channels)]
    ;;I go ahead and create a contiguous matrix here because we will iterate over this many times
    (convolution-sequence input-matrix output-width output-height
                          conv-layer-config)))

(defn get-gradient-convolution-sequence
  "returns [conv-sequence input-mat-view] for the backpass steps of nn layers
using convolutional steps"
  [{:keys [^long width ^long height ^long k-width ^long k-height
           ^long padx ^long pady ^long stride-w ^long stride-h
           ^long num-channels] :as conv-layer-config}]
  (let [input-matrix (create-padded-matrix conv-layer-config)
        input-mat-view (m/submatrix input-matrix
                                    [[pady height]
                                     [(* padx num-channels) (* width num-channels)]])
        output-width (get-padded-strided-dimension width padx k-width stride-w)
        output-height (get-padded-strided-dimension height pady k-height stride-h)
        kernel-stride (* k-width num-channels)]
    [(convolution-sequence input-matrix output-width output-height
                           conv-layer-config)
     input-mat-view
     input-matrix]))


(defn core-matrix-unroll-input!
  [this input]
  (let [conv-layer-config (:conv-layer-config this)
        input-data (or (:input-data this)
                       (let [[conv-rows input-view padded-backing-matrix]
                             (get-gradient-convolution-sequence conv-layer-config)]
                         [(into [] conv-rows) (m/as-vector input-view) padded-backing-matrix]))
        [conv-rows input-view padded-backing-matrix] input-data

        ;;If there is any padding, then we have to clear the backing matrix
        _ (when-not (= (m/shape input) (m/shape (m/as-vector padded-backing-matrix)))
            (m/scale! padded-backing-matrix 0.0))
        _ (m/assign! input-view input)
        packed-conv-matrix  (if-let [conv-matrix (:packed-conv-matrix this)]
                              (let [packed-rows (vec (m/rows conv-matrix))
                                    row-count (count packed-rows)]
                                (loop [row-idx 0]
                                  (when (< row-idx row-count)
                                    (m/assign! (packed-rows row-idx) (conv-rows row-idx))
                                    (recur (inc row-idx))))
                                conv-matrix)
                              (b/array (seq conv-rows)))]
    (assoc this :input-data input-data
           :packed-conv-matrix packed-conv-matrix)))

#?(:cljs (defn unroll-input!
           [this input]
           (core-matrix-unroll-input! this input)))

#?(:clj (defn unroll-input!
          [this input]
          (let [conv-layer-config (:conv-layer-config this)
                input-data (or (:input-data this)
                               (let [[conv-rows input-view padded-backing-matrix]
                                     (get-gradient-convolution-sequence conv-layer-config)]
                                 [(into [] conv-rows) (m/as-vector input-view)
                                  padded-backing-matrix]))
                [conv-rows input-view padded-backing-matrix] input-data

                packed-conv-matrix (or (:packed-conv-matrix this)
                                       (b/new-array [(count conv-rows)
                                                     (m/ecount (first conv-rows))]))
                {:keys [^long width ^long height
                        ^long k-width ^long k-height
                        ^long padx ^long pady
                        ^long stride-w ^long stride-h
                        ^long num-channels]} conv-layer-config
                input-ary (mp/as-double-array input)
                packed-ary (mp/as-double-array packed-conv-matrix)
                this (assoc this
                            :input-data input-data
                            :packed-conv-matrix packed-conv-matrix)]
            (if (and input-ary packed-ary)
              (do
               (ConvOps/unrollInput width height k-width k-height
                                    padx pady stride-w stride-h
                                    num-channels input-ary packed-ary)
               this)
              (core-matrix-unroll-input! this input)))))


(defn convolution-forward!
  [this weights bias packed-unrolled-input]
  (if (and (blas/supports-blas? weights)
           (blas/supports-blas? packed-unrolled-input))
    (let [output (or (:output-mat this)
                     (b/new-array [(m/row-count packed-unrolled-input)
                                   (m/row-count weights)]))]
      (m/assign! output bias)
      (blas/gemm! output false true 1.0
                  packed-unrolled-input weights
                  1.0)
      (assoc this :output-mat output
             :output (m/as-vector output)))
    (let [weights-t (m/transpose weights)
          output (m/inner-product packed-unrolled-input weights-t)]
      (m/add! output bias)
      (assoc this :output (m/as-vector output)))))



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Convolution backward pass utility functions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(defn convolution-copying-roll
  [packed-gradient gradient-view-rows gradient-view]
  (doall (map m/assign! gradient-view-rows packed-gradient))
  (m/array :vectorz (m/as-vector gradient-view)))

(defn non-blas-convolution-backward!
  [weights bias input output-gradient {:keys [conv-layer-config] :as this}]
  (let [packed-conv-matrix (:packed-conv-matrix this)
        [conv-rows input-view padded-backing-matrix] (:input-data this)
        input-gradient (or (:input-gradient this)
                           (b/array (repeat (first (m/shape input)) 0.0)))
        ;;Note that at this point packed-conv-matrix holds the input exploded into convolutions
        ;;We are going to use the padded backing matrix as an accumulator
        _ (m/scale! padded-backing-matrix 0.0)
        ;;the backward pass is optimized to only minimally access conv-rows.
        ;;This is because they
        ;;are views of views (m/as-vector (m/submatrix ...)) which implies a significant
        ;;performance hit at this time.
        {:keys [^long width ^long height
                ^long k-width ^long k-height
                ^long padx ^long pady
                ^long stride-w ^long stride-h
                ^long num-channels]}  conv-layer-config
        ;;We use an accumulator here because the input-gradient-conv-sequence
        ;;could be a set of view-on-views or some other deep abstraction.
        ;;Using an accumulator is a significant performance helper during the tight
        ;;per-kernel gradient accumulation
        ^long kernel-count (first (m/shape weights))
        output-row-count (quot (long (first (m/shape output-gradient))) kernel-count)
        accum (b/zero-array [(* k-width k-height num-channels)])
        weight-gradient (or (:weight-gradient this)
                            (b/new-array (m/shape weights)))
        bias-gradient (or (:bias-gradient this)
                          (b/new-array [kernel-count]))]
    (loop [idx 0]
      (when (< idx output-row-count)
        (let [input-gradient-row (m/get-row conv-rows idx)
              input-row (m/get-row packed-conv-matrix idx)]
          (m/assign! accum input-gradient-row)
          (loop [kern-idx 0]
            (when (< kern-idx kernel-count)
              (let [weight-row (m/get-row weights kern-idx)
                    weight-gradient-row (m/get-row weight-gradient kern-idx)
                    output-gradient-offset (+ (* idx kernel-count)
                                              kern-idx)
                    ^double gradient (m/mget output-gradient output-gradient-offset)
                    ^double bias-gradient-val (m/mget bias-gradient kern-idx)]
                (m/add-scaled! accum weight-row gradient)
                (m/add-scaled! weight-gradient-row input-row gradient)
                (m/mset! bias-gradient kern-idx (+ gradient bias-gradient-val)))
              (recur (inc kern-idx))))
          (m/assign! input-gradient-row accum))
        (recur (inc idx))))
    (m/assign! input-gradient (m/as-vector input-view))
    (assoc this
           :input-gradient input-gradient
           :weight-gradient weight-gradient
           :bias-gradient bias-gradient)))

(defn core-matrix-roll-input-gradient!
  [this input-gradient packed-conv-matrix]
  (let [input-data (:input-data this)
        [conv-rows input-view padded-backing-matrix] input-data]
    ;;Summation into the input gradient.  This *cannot* be parallelized at all at the moment,
    ;;probably some perf loss.  What you don't see is that conv rows are views into the padded
    ;;backing matrix.  In essence we are rolling back up the unrolled input
    (m/mset! padded-backing-matrix 0.0)
    (let [row-count (m/row-count packed-conv-matrix)
          packed-conv-matrix packed-conv-matrix]
      (loop [row-idx 0]
        (when (< row-idx row-count)
          (let [input-row (m/get-row packed-conv-matrix row-idx)
                output-row (conv-rows row-idx)]
            (m/add! output-row input-row))
          (recur (inc row-idx)))))
    ;;This assignment is a nop in the case where input-gradient is input view
    (m/assign! input-gradient input-view)
    this))

#?(:cljs (defn roll-input-gradient!
           [this input-gradient packed-conv-matrix]
           (core-matrix-roll-input-gradient! this input-gradient packed-conv-matrix)))

#?(:clj (defn roll-input-gradient!
          [this input-gradient packed-conv-matrix]
          (let [gradient-ary (mp/as-double-array input-gradient)
                packed-ary (mp/as-double-array packed-conv-matrix)
                conv-layer-config (:conv-layer-config this)
                {:keys [^long width ^long height
                        ^long k-width ^long k-height
                        ^long padx ^long pady
                        ^long stride-w ^long stride-h
                        ^long num-channels]} conv-layer-config]
            (if (and gradient-ary packed-ary)
              (do
                (java.util.Arrays/fill ^doubles gradient-ary 0.0)
                (ConvOps/rollInput width height k-width k-height padx pady
                                   stride-w stride-h num-channels
                                   gradient-ary packed-ary)
                this)
              (core-matrix-roll-input-gradient! this input-gradient packed-conv-matrix)))))


(defn blas-convolution-backward!
  [weights bias input output-gradient {:keys [conv-layer-config] :as this}]
  (let [input-data (:input-data this)
        [conv-rows input-view padded-backing-matrix] input-data
        packed-input-matrix (:packed-conv-matrix this)
        {:keys  [^long width ^long height ^long k-width ^long k-height
                 ^long padx ^long pady ^long stride-w ^long stride-h
                 ^long num-channels]} conv-layer-config
        output-width (get-padded-strided-dimension width padx k-width stride-w)
        output-height (get-padded-strided-dimension height pady k-height stride-w)
        n-kernels (m/row-count weights)
        n-output-pixels (* output-height output-width)

        output-gradient-matrix (or (:output-gradient-matrix this)
                                   (b/new-array [n-output-pixels n-kernels]))

        input-stride (* k-width k-height num-channels)
        ;;Note that in the non-padded case there is no need to have separate
        ;;input-view and input-gradient objects.  The gradient we pass upstream
        ;;does need to be dense, however.
        input-gradient (or (:input-gradient this)
                           (b/new-array (m/shape input)))
        weight-gradient (or (:weight-gradient this)
                            (b/new-array (m/shape weights)))
        bias-gradient (or (:bias-gradient this)
                          (b/new-array [n-kernels]))
        vector-of-ones (or (:vector-of-ones this)
                           (b/array (repeat n-output-pixels 1.0)))]
    (m/assign! (m/as-vector output-gradient-matrix) output-gradient)

    (blas/gemm! weight-gradient true false 1.0 output-gradient-matrix packed-input-matrix
                1.0)
    (blas/gemm! packed-input-matrix false false 1.0 output-gradient-matrix weights
                0.0)
    (blas/gemv! bias-gradient true 1.0 output-gradient-matrix vector-of-ones
                1.0)

    (assoc (roll-input-gradient! this input-gradient packed-input-matrix)
           :output-gradient-matrix output-gradient-matrix
           :input-gradient input-gradient
           :bias-gradient bias-gradient
           :weight-gradient weight-gradient
           :vector-of-ones vector-of-ones)))

(defn convolution-backward!
  ([this input output-gradient]
   (let [weights (:weights this)
         packed-conv-matrix (:packed-conv-matrix this)
         bias (:bias this)]
    (if (and (blas/supports-blas? (:packed-conv-matrix this))
             (blas/supports-blas? weights)
             (blas/supports-blas? bias))
      (blas-convolution-backward! weights bias input output-gradient this)
      (non-blas-convolution-backward! weights bias input output-gradient this)))))



;; Explanation of the the conv-layer algorithm:
;;
;; Forward: Take the input which is expected to be a single interleaved vector of data
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
;; This means the convolution step is:
;; (m/matrix-multiply convolved-rows (m/transpose weights))
;; The result is an interleaved matrix that looks like:
;; [[cr1k1 cr1k2 cr1k3 cr1k4] (repeats for num convolved rows...)].
;; Note that simply calling as-vector means our output is in the exact same format
;; as our input where each convolution kernel output plane is interleaved.  It also means
;; we can use our linear layer with no changes directly after a convolution layer.
;;


#?(:cljs (register-module cortex.impl.layers.Convolutional))
(defrecord Convolutional [weights bias weight-gradient bias-gradient
                          conv-layer-config]
    cp/PModule
    (cp/calc [this input]
      (let [this (unroll-input! this input)
            packed-conv-matrix (:packed-conv-matrix this)]
        (convolution-forward! this weights bias packed-conv-matrix)))

    (cp/output [m]
      (:output m))

    cp/PNeuralTraining
    (forward [this input]
      (cp/calc this input))

    (backward [this input output-gradient]
      (convolution-backward! this input output-gradient))

    (input-gradient [this]
      (:input-gradient this))

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


(defn max-pooling-forward-default!
  "The forward pass writes to two things; output and output-indexes.  The indexes
record which item from the input row we actually looked at.  Returns
[output output-indexes].  Note that this does not change the number of channels
so it needs to remember the max per-channel."
  [input output output-indexes
   {:keys [^long height ^long width
           ^long padx ^long pady
           ^long k-width ^long k-height
           ^long stride-w ^long stride-h
           ^long num-channels] :as conv-layer-config}]
  ;;Each input row has k-width*num-channels*k-height in it.
  ;;Each output index gets num-channels written to it.
  (let [conv-sequence (convolution-operation-sequence conv-layer-config)
        height (long height)
        width (long width)
        padx (long padx)
        pady (long pady)
        k-width (long k-width)
        k-height (long k-height)
        stride-w (long stride-w)
        stride-h (long stride-h)
        num-channels (long num-channels)
        output-width (get-padded-strided-dimension width padx k-width stride-w)]
    (doseq [conv-item conv-sequence]
      (let [{:keys [^long input-x ^long input-y ^long conv-x ^long conv-y
                    ^long output-x ^long output-y]} conv-item
            input-x (long input-x)
            input-y (long input-y)
            conv-x (long conv-x)
            conv-y (long conv-y)
            output-x (long output-x)
            output-y (long output-y)
            output-offset (* num-channels (+ (* output-y output-width) output-x))]
        (loop [conv-x conv-x]
          (when (< conv-x k-width)
            (let [input-offset-x (+ input-x conv-x)
                  valid-input? (and (>= input-y 0)
                                    (< input-y height)
                                    (>= input-offset-x 0)
                                    (< input-offset-x width))
                  kernel-index (+ (* conv-y k-width) conv-x)
                  input-offset (* num-channels (+ (* input-y width) input-offset-x))]
              (loop [chan 0]
                (when (< chan num-channels)
                  (let [input-val (double (if valid-input?
                                            (m/mget input (+ input-offset chan))
                                            0.0))
                        output-offset (+ output-offset chan)
                        existing-value (double (m/mget output output-offset))]
                    (when (or (= kernel-index 0)
                              (> input-val existing-value))
                      (m/mset! output output-offset input-val)
                      (m/mset! output-indexes output-offset kernel-index)))
                  (recur (inc chan)))))
            (recur (inc conv-x))))))
    [output output-indexes]))


#?(:cljs(defn max-pooling-forward!
          [input output output-indexes conv-config]
          (max-pooling-forward-default! input output output-indexes conv-config))

   :clj(defn max-pooling-forward!
         [input output output-indexes conv-config]
         (let [input-ary (mp/as-double-array input)
               output-ary (mp/as-double-array output)
               output-indexes-ary (mp/as-double-array output-indexes)
               {:keys [width height k-width k-height padx pady stride-w stride-h
                       num-channels]} conv-config]
           (if (and input-ary output-ary output-indexes-ary)
             (ConvOps/maxPooling width height k-width k-height padx pady
                                 stride-w stride-h num-channels
                                 input-ary output-ary output-indexes-ary)
             (max-pooling-forward-default! input output output-indexes conv-config)))))


(defn max-pooling-backward!
  "Calculates the input gradient using the inverse of the convolution step
combined with the output gradient and the output indexes which tell you which kernel
index the output came from."
  [output-gradient output-indexes input-gradient
   {:keys [^long width ^long height
           ^long k-width ^long k-height
           ^long padx ^long pady
           ^long stride-w ^long stride-h
           ^long num-channels] :as conv-layer-config}]
  (m/mset! input-gradient 0.0)
  (let [output-width (get-padded-strided-dimension width padx k-width stride-w)
        output-height (get-padded-strided-dimension height pady k-height stride-h)
        num-pixels (* output-width output-height)]
    (loop [pixel 0]
      (when (< pixel num-pixels)
        (let [output-offset (* pixel num-channels)
              output-x (rem pixel output-width)
              output-y (quot pixel output-width)]
          (loop [chan 0]
            (when (< chan num-channels)
              (let [output-offset (+ output-offset chan)
                    kernel-index (long (m/mget output-indexes output-offset))
                    conv-x (rem kernel-index k-width)
                    conv-y (quot kernel-index k-width)
                    input-x (- (+ (* output-x stride-w) conv-x) padx)
                    input-y (- (+ (* output-y stride-h) conv-y) pady)]
                (when (and (>= input-y 0)
                           (< input-y height)
                           (>= input-x 0)
                           (< input-x width))
                  (let [^double output-gradient-value (m/mget output-gradient output-offset)
                        input-offset (+ (* num-channels (+ (* input-y width)
                                                           input-x))
                                        chan)
                        ^double input-gradient-value (m/mget input-gradient input-offset)]
                    (m/mset! input-gradient input-offset (+ input-gradient-value
                                                            output-gradient-value)))))
              (recur (inc chan)))))
        (recur (inc pixel))))))


;;Max pooling layer.  There are other pooling layer types (average,a sochiastic)
;;that may be implemented later but for now we only need max pooling.
#?(:cljs (register-module cortex.impl.layers.Pooling))
(defrecord Pooling [output output-indexes input-gradient conv-layer-config]
  cp/PModule
  (cp/calc [this input]
    (max-pooling-forward! input output output-indexes conv-layer-config))

  (cp/output [m]
    (:output m))

  cp/PNeuralTraining
  (forward [this input]
    (cp/calc this input)
    this)

  (backward [this input output-gradient]
    (max-pooling-backward! output-gradient output-indexes
                           input-gradient conv-layer-config)
    this)

  (input-gradient [this]
    (:input-gradient this)))
