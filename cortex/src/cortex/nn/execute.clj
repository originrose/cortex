(ns cortex.nn.execute
  "Executing the graph means training or inference.  The goal is to allow both imperative/effectful implementations
and pure functional implementations but to abstract common details of training or execution into
one place written in such a way that someone can affect the behavior of various implementations and design
new execution strategies (like parameter sharing) at least partially without needing to work withing a specific
implementation.  It is important to realize that training the network means essentially a transformation from
layer-graph -> layer-graph via some training process."
  (:require [cortex.nn.traverse :as traverse]
            [cortex.nn.layers :as layers]))


(defprotocol PExecutionContext
  "A specific execution context implements all of the specific functionality of the network such as
the nodes, loss functions, optimization engines, and various other details."
  (bind-to-network [context built-network traverse batch-size]
    "Bind an execution context to a network.  This should return a new network with any specific
information the context needs embedded in it.")
  (train-batch-sequence [context built-network dataset-epoch]
    "Return a lazy sequence of progressively better trained built-networks, one for each batch.")
  (infer-batch-sequence [context built-network dataset-epoch]
    "Return a lazy sequence of maps of output-name->double-array-seq")
  (save-to-network [context built-network]
    "Return a new network without context information and with any persistent information
(like parameters) updated.  This may be called multiple times during the training process.")
  )
