(ns think.compute.loss
  "Loss implementations across the compute system."
  (:require [think.compute.math :as math]
            [clojure.core.matrix :as m]
            [think.compute.driver :as drv]))


(defmulti compute-loss-gradient
  "Compute the loss gradient and place into gradient.  There is a companion
multimethod defined in cortex.loss in order to compute the loss.  The loss
is calculated uniformly on the cpu however and thus does not require
specific compute implementations."
  (fn [loss-fn backend v target gradient]
    (get loss-fn :type)))


(defmethod compute-loss-gradient :mse-loss
  [loss-fn backend v target gradient]
  (let [stream (drv/get-stream backend)
        [batch-size output-size] (math/batch-shape v)
        alpha (/ 2.0 (double output-size))]
    (math/subtract stream
                   alpha (math/device-buffer v)
                   alpha (math/device-buffer target)
                   (math/device-buffer gradient))))


(defn- calculate-cross-entropy-gradient
  [backend v target gradient]
  (let [stream (drv/get-stream backend)
        elem-count (m/ecount gradient)
        alpha 1.0]
    (math/subtract stream
                   alpha (math/device-buffer v)
                   alpha (math/device-buffer target)
                   (math/device-buffer gradient))))


(defmethod compute-loss-gradient :softmax-loss
  [loss-fn backend v target gradient]
  (calculate-cross-entropy-gradient backend v target gradient))
