# Written Character Recognition

Classifying mnist digits using a convolutional neural network.

## Usage

Running the example will train the network from scratch and evaluate the fraction that was correct.

`lein run`

```
Training convnet on MNIST from scratch.
epoch 1  mse-loss: 0.00880103526187728
epoch 2  mse-loss: 0.003686973971882442
Network score: 0.976300
Network mse-score 0.00368697
```

_Note that the network training uses core.matrix as the backend. For another example with optimizations for speed, look at:_

*  think.compute.verify.nn.mnist that uses cpu-compute
*  the suite classification example that uses gpu-compute
