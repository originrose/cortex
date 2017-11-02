(ns cortex.compute.nn.activations
  "High level implemenations of activations that work across all backends"
  (:require [clojure.core.matrix :as m]
            [cortex.tensor :as tensor]
            [think.datatype.core :as dtype]))


;;;; Forward

(defn logistic [input output]
  (tensor/unary-op! output 1.0 input :logistic))

(defn tanh [input output]
  (tensor/unary-op! output 1.0 input :tanh))

(defn relu [input output]
  (tensor/binary-op! output 1.0 input 0 0 :max))

(defn swish
  "sigmoid(x)*x"
  [input output]
  (tensor/unary-op! output 1.0 input :logistic)
  (tensor/binary-op! output 1.0 input 1.0 output :*))

;;; Backwards

(defn default-gradient
  "Provides the default gradient for tanh, logistic, and relu"
  [input-gradient output-gradient output act-type]
  (tensor/activation-gradient! input-gradient output-gradient output act-type))

(defn logistic-gradient [input-gradient output-gradient output]
  (default-gradient input-gradient output-gradient output :logistic))

(defn tanh-gradient [input-gradient output-gradient output]
  (default-gradient input-gradient output-gradient output :tanh))

(defn relu-gradient [input-gradient output-gradient output]
  (default-gradient input-gradient output-gradient output :relu))

(defn swish-gradient
  "(fx + sigm *(1 -fx)) * output-grad - where fx = sigm(x) * x"
  [input-gradient output-gradient output]
  (let [fx (tensor/new-tensor (m/shape output) :datatype (dtype/get-datatype output))
        sigm (tensor/new-tensor (m/shape output) :datatype (dtype/get-datatype output))]
    ;;fx
    (tensor/unary-op! fx 1.0 output :logistic)
    (tensor/binary-op! fx 1.0 fx 1.0 output :*)
    ;; sigm
    (tensor/unary-op! sigm 1.0 output :logistic)
    ;; (fx + sigm*(1-fx)
    (tensor/binary-op! input-gradient 1.0 1.0 1.0 fx :-)
    (tensor/binary-op! input-gradient 1.0 input-gradient 1.0 sigm :*)
    (tensor/binary-op! input-gradient 1.0 input-gradient 1.0 fx :+)
    ;; mult to the output-grad
    (tensor/binary-op! input-gradient 1.0 input-gradient 1.0 output-gradient :*)
    input-gradient))
