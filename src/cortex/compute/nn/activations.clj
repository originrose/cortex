(ns cortex.compute.nn.activations
  "High level implemenations of activations that work across all backends"
  (:refer-clojure :exclude [ * - + >])
  (:require [cortex.tensor :as tensor]
            [cortex.tensor.operations :as tops :refer [* - + > new-tensor exp where]]))

;;;; Forward

(defn logistic [input output]
  (tops/logistic output input))

(defn tanh [input output]
  (tops/tanh output input))

(defn relu [input output]
  (tops/max output input 0))

(defn swish
  "sigmoid(x)*x"
  [input output]
  (-> (tops/logistic output input)
      (* input)))

(def SELU_ALPHA 1.6732632423543772848170429916717)
(def SELU_LAMBDA 1.0507009873554804934193349852946)

(defn selu
  "lambda*x for x > 0 and lambda * ((alpha * exp(x)) - alpha) for x <=0"
  [input output]
  (where output
         (> (new-tensor input) input 0)
         ; lambda*x for x > 0
         (* (new-tensor input) input SELU_LAMBDA)
         ;  lambda * ((alpha * exp(x)) - alpha) for x <=0
         (-> (exp (new-tensor input) input)
             (* SELU_ALPHA)
             (- SELU_ALPHA)
             (* SELU_LAMBDA))))

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
  (let [fx (-> (tops/logistic (new-tensor output) output)
               (* output))
        sigm (tops/logistic (new-tensor output) output)]

    ;; (fx + sigm*(1-fx)
    (-> (- input-gradient 1.0 fx)
        (* sigm)
        (+ fx)
        ;; mult to the output-grad
        (* output-gradient))))

(defn selu-gradient
  "lambda for x > 0 and lambda * alpha exp(x) for x <= 0"
  [input-gradient output-gradient output]
  (-> (where input-gradient
              (> (new-tensor output) output 0)
              ;; "lambda for x > 0
              (+ (new-tensor output) SELU_LAMBDA)
              ; "lambda for x > 0
              (-> (exp (new-tensor output) output)
                  (* SELU_ALPHA)
                  (* SELU_LAMBDA)))
      (* output-gradient)))
