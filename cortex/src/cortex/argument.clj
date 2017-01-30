(ns cortex.argument
  "Cortex has at this point 2 major extension mechanisms; one for loss functiosn
and one for nodes.  Nodes and loss functions both can be modelled as functions taking
named arguments with nodes taking an extra set of input and output implicit arguments
that are defined by the graph structure.  Argument's that operate on a node's output
or a parameter buffer (either external or specific to this item) may have a gradient
buffer associated with them (indicated by a gradients? member of the argument's metadata).

Arguments have metadata associated with them.  This metadata explains to the system how to
use the argument but it isn't strictly necessary.
In general any functions required for the argument (such as initialization or shape definition)
cannot be specified in the graph itself because the graph itself needs to be serialized to disk.
Thus support functions are indicated in general by keyword.

At runtime the system sees arguments that have been flattened in this form:

(merge metadata item-argument-definition buffer-info)

This avoids various subsystems from needing to lookup information from disparate parts
of the graph structure.

Arguments currently can have 5 different sources:
1.  A node's output (if the node has exactly 1 output)
2.  An external entity's parameter buffer.
3.  A data stream
4.  A parameter buffer.  Data tracked by the implementation with an initializer.  Parameter
    buffers furthermore may have gradients associated with them.
5.  An augmented stream.  A pure transformation from a data stream to another stream."
  (:require [clojure.core.matrix :as m]
            [cortex.util :refer [arg-list->arg-map merge-args]]))


(defmulti get-argument-metadata
  "Return any metadata associated with the argument."
  :type)


(defn ->node-output-arg
  "Bind to a graph node's output.  Error results if node has multiple outputs.
Defaults to assuming the function produces gradients for this argument."
  [node-id & args]
  (merge-args
   {:type :node-output
    :node-id node-id}
   args))


(defmethod get-argument-metadata :node-output
  [argument]
  {:gradients? true})


(defn ->node-param-arg
  "Bind to a graph node's parameter buffer."
  [node-id param-key & args]
  (merge-args
   {:type :node-parameter
    :node-id node-id
    :parameter param-key}
   args))


(defmethod get-argument-metasdata :node-parameter
  [argument]
  {:gradients? true})


(defn ->stream-arg
  "Bind to a data stream."
  [stream-name & args]
  (merge-args
   {:type :stream
    :stream stream-name}
   args))


(defmethod get-argument-metadata :stream
  [argument]
  {})


(defn ->parameter-arg
  "Bind to a parameter buffer.  Note that if a buffer is specified
then the initializer for this argument is ignored and the buffer is used."
  [shape-fn-id initializer & args]
  (merge-args
   {:type :parameter
    :shape-fn shape-fn-id
    :initialization initializer}
   args))


(defmethod get-argument-metadata :parameter
  [argument]
  {})


(defn ->argumented-stream-arg
  [stream-name augmentation & args]
  (merge-args
   {:type :stream-augmentation
    :stream stream-name
    :augmentation augmentation}
   args))


(defmethod get-argument-metadata :stream-agumentation
  [parameter]
  {})
