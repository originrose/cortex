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


(defn ->node-argument-arg
  "Bind to a graph node's argument"
  [node-id arg-key & args]
  (merge-args
   {:type :node-argument
    :node-id node-id
    :argument arg-key}
   args))


(defmethod get-argument-metadata :node-argument
  [argument]
  {})


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
then the initializer for this argument is ignored and the buffer is used.
the shape function is a keyword function where the namespace, fn-name and
possible arguments are encoded in map
{:fn function
:args args}.
see cortex.keyword-fn namespace."
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
  "Augmentation is a keyword function.  See cortex.keyword-fn.
The augmentation points to another argument which must of type stream."
  [stream-name augmentation & args]
  (merge-args
   {:type :stream-augmentation
    :stream stream-name
    :augmentation augmentation}
   args))


(defn augmented-stream-arg->id
  "Note that this relies on the argument the augment-arg points to
being resolved to the stream it refers to and this stream being assoc'd
back into the stream augmentation argument."
  [argument]
  (select-keys argument [:stream :augmentation]))


(defmethod get-argument-metadata :stream-augmentation
  [parameter]
  {})


(defn set-arg-node-output
  [node arg-name node-id]
  (update node arg-name
          #(merge %
                  {:type :node-output
                   :node-id node-id})))


(defn set-arg-node-argument
  [node arg-name node-id node-arg-name]
  (update node arg-name
          #(merge %
                  {:type :node-argument
                   :node-id node-id
                   :argument node-arg-name})))

(defn set-arg-stream
  [node arg-name stream-name]
  (update node arg-name
          #(merge %
                  {:type :stream
                   :stream stream-name})))
