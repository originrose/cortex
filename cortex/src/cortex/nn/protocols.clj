(ns cortex.nn.protocols)


(defprotocol PExecutionContext
  "A specific execution context implements all of the specific functionality of the network such as
the nodes, loss functions, optimization engines, and various other details.
There is a concept of a batch map sequence which is a sequence of maps of stream-names to
batches of data.  This is the format produced by the dataset abstraction but it isn't strictly
necessary to use the dataset abstraction in order to train or infer."
  (bind-to-network [context built-network options]
    "Bind an execution context to a network.  This should return a new network with any specific
information the context needs embedded in it.  The network contains at least:
{:layer-graph
 :traversal
 :batch-size}")
  (train-batch-sequence [context built-network batch-map-sequence options]
    "Return a sequence of progressively better trained built-networks, one for each batch.")
  (infer-batch-sequence [context built-network batch-map-sequence options]
    "Return a sequence of maps of node-id->double-array-seq.  Use dataset/batch-sequence-columnar in order
to transform sequence into specific sequences.")
  (save-to-network [context built-network options]
    "Return a new network without context information and with any persistent information
(like parameters) updated.  This may be called multiple times during the training process.
Options is map that may contain:
save-gradients? - save the gradients *and* the io buffers.")

  ;;Test/verification interfaces
  (forward-backward [context built-network
                     stream->input-map
                     node-id->output-gradient-map]
    "Test interface - Run the network forward and backward using these inputs and output-gradients.")
  (forward-backward-loss [context built-network
                          stream->input-map
                          node-id->loss-function-answer-map]
    "Run network forward and backward like 'forward-backward' but also calculate numeric
gradients w/r/t the loss function and the provided answer.  This allows for gradient
checking.  The data should be saved back to the network after the passes"))
