(ns cortex.nn.impl
  "Implementation helpers to aid implementing neural network cortex protocols
or specific neural network layers"
  (:require [cortex.nn.layers :as layers]
            [clojure.core.matrix.macros :refer [c-for]]))


(defmacro convolution-outer-kernel
  [conv-desc & body]
  `(let [~'conv-desc ~conv-desc
         ~'output-width (long (:output-width ~'conv-desc))
         ~'output-height (long (:output-height ~'conv-desc))
         ~'num-in-channels (long (:input-channels ~'conv-desc))
         ~'num-out-channels (long (:output-channels ~'conv-desc))
         ~'input-width (long (:input-width ~'conv-desc))
         ~'input-height (long (:input-height ~'conv-desc))
         ~'input-planar-stride (* ~'input-width ~'input-height)
         ~'output-planar-stride (* ~'output-width ~'output-height)
         ~'kernel-width (long (:kernel-width ~'conv-desc))
         ~'kernel-height (long (:kernel-height ~'conv-desc))
         ~'output-channel-stride (* ~'kernel-width ~'kernel-height)
         ~'output-column-stride (* ~'output-channel-stride ~'num-in-channels)
         ~'stride-y (long (:stride-y ~'conv-desc))
         ~'stride-x (long (:stride-x ~'conv-desc))
         ~'pad-x (long (:pad-x ~'conv-desc))
         ~'pad-y (long (:pad-y ~'conv-desc))
         ~'min-x (- 0 ~'pad-x)
         ~'min-y (- 0 ~'pad-y)
         ~'max-x (+ ~'input-width ~'pad-x)
         ~'max-y (+ ~'input-height ~'pad-y)]
     (c-for
      [~'chan 0 (< ~'chan ~'num-in-channels) (inc ~'chan)]
      (let [~'chan-input-offset (* ~'chan ~'input-planar-stride)
            ~'chan-output-offset (* ~'chan ~'output-planar-stride)]
        (c-for
         [~'out-y 0 (< ~'out-y ~'output-height) (inc ~'out-y)]
         (let [~'input-rel-y (- (* ~'out-y ~'stride-y) ~'pad-y)]
           (c-for
            [~'out-x 0 (< ~'out-x ~'output-width) (inc ~'out-x)]
            (let [~'input-rel-x (- (* ~'out-x ~'stride-x) ~'pad-x)]
              ~@body))))))))


(defmacro in-bounds?
  "is value within the range of [min-val, max-val)"
  [value min-val max-val]
  `(and (>= ~value ~min-val)
        (< ~value ~max-val)))


(defmacro convolution-roll-unroll-inner-kernel
  [& body]
  `(let [~'chan-conv-offset (* ~'chan ~'output-channel-stride)
         ~'output-offset (+ (* ~'out-y ~'output-width)
                            ~'out-x)]
     (c-for
      [~'k-y 0 (< ~'k-y ~'kernel-height) (inc ~'k-y)]
      (c-for
       [~'k-x 0 (< ~'k-x ~'kernel-width) (inc ~'k-x)]
       (let [~'input-x (+ ~'input-rel-x ~'k-x)
             ~'input-y (+ ~'input-rel-y ~'k-y)
             ~'output-conv-addr (+ (* ~'output-offset
                                      ~'output-column-stride)
                                   ~'chan-conv-offset
                                   (* ~'k-y ~'kernel-width)
                                   ~'k-x)
             ~'input-addr  (+ (* ~'input-y ~'input-width)
                              ~'input-x
                              ~'chan-input-offset)
             ~'input-valid? (and (in-bounds? ~'input-x 0 ~'input-width)
                                 (in-bounds? ~'input-y 0 ~'input-height))
             loop-valid?# (and (in-bounds? ~'input-x ~'min-x ~'max-x)
                               (in-bounds? ~'input-y ~'min-y ~'max-y))]
         (when loop-valid?#
           ~@body))))))
