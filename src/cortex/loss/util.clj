(ns cortex.loss.util
  (:require [cortex.loss.core :as loss]
            [cortex.graph :as graph]))

(defn generate-loss-term-stream-definitions
  "Some loss terms have a constraint that the stream ecount must match the node's
output ecount.  For those loss terms generating a stream definition is possible."
  [graph loss-term]
  (let [arguments (graph/get-node-arguments loss-term)
        node-output (-> (filter #(= :node-output (get % :type)) arguments)
                        first)
        stream-input (-> (filter #(= :stream (get % :type)) arguments)
                         first)]
    (if (and node-output stream-input)
      (let [stream-name (get stream-input :stream)
            node-id (get node-output :node-id)
            output-node (graph/get-node graph node-id)
            output-size (graph/node->output-size output-node)]
        ;;Return one stream definition and one output size
        [[stream-name [output-size]]])
      [])))


(defmulti create-compute-loss-term
  "Multi method to allow pluggable loss terms.  Note that formally defined parameters are
taken care of for you."
  (fn [backend network loss-term batch-size]
    (:type loss-term)))


(defprotocol PComputeLoss
  "Compute implementation to compute loss gradient for a given loss term.  Gradient
buffer is expected to be entirely overwritten by operation."
  (compute-loss-gradient [loss-term buffer-map]))


(defn generic-loss-term
  "Generate a generic loss term from a loss type"
  [loss-type]
  {:type loss-type
   :lambda (loss/get-loss-lambda {:type loss-type})})
