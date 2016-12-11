# suite-classification

This example shows of some use of the continual training with a convolutional neural network with visualization as well as inference with a random image from the dataset. It uses the classic MNIST handwritten digit corpus. The training of the dataset using augmentation, to slightly change the image, for continual training.

## Usage

You will need to have CUDA installed to be able to run this example. Please follow the directions in the README of the main project for GPU Compute installation instructions.

_If you have CUDA-8.0 installed before you run the example, please uncomment the needed lib `[org.bytedeco.javacpp-presets/cuda "8.0-1.2"]` in the project.clj_

Run `lein run`

```
13:02 $ lein run
checking that we have produced all images
building dataset
Figwheel: Starting server at http://0.0.0.0:3449
Figwheel: Watching build - dev
Figwheel: Cleaning build - dev
Compiling "resources/public/js/app.js" from ["cljs"]...
Successfully compiled "resources/public/js/app.js" in 3.552 seconds.
```

You will see output about building the image sets, then you will see figwheel starting and compiling some ClojureScript files. This is for the confusion matrix visualization. After you see `Successfully compiled`, go ahead and open a web page at [http://localhost:8090/](http://localhost:8090/).  This will load the confusion matrix visualization.
  
Notice the log in the console

```
Loss for epoch 0: [0.010127717832745782]
Loss for epoch 1: [0.009369591868873164]
Saving network
Loss for epoch 2: [0.01106402198376483]
Loss for epoch 3: [0.008233267247032423]
Saving network
Loss for epoch 4: [0.008045399444719475]
Saving network
Loss for epoch 5: [0.007743433619827841]
Saving network
Loss for epoch 6: [0.009741945370601088]
Loss for epoch 7: [0.007961874145386162]
Loss for epoch 8: [0.008225386161276766]
Loss for epoch 9: [0.008363428905793118]
Loss for epoch 10: [0.0077346386730130215]
Saving network
```

Each time it saves the network, it exports it to an external nippy file. `trained-network.nippy`

When you think it's run long enough, go ahead a stop the process.
Get a new REPL up in the core namespace 
Run the `label-one` function

```
suite-classification.core> (clojure.pprint/pprint (label-one))
{:probability-map
 {"9" 0.9999990463256836,
  "3" 8.426678244077834E-10,
  "4" 9.943216809915612E-7,
  "8" 1.5535771780150753E-8,
  "7" 4.730325020574355E-10,
  "5" 8.71147338293854E-12,
  "6" 5.481395841237281E-15,
  "1" 5.038783518034051E-13,
  "0" 1.0979338098404678E-12,
  "2" 6.575276781384254E-12},
 :classification "9"}
 ```
 You will see the results of a randomly selected handwritten digit being classified using the last exported network. Also you will see the popup of the actual image that it is trying to classify.




## License

Copyright © 2016 FIXME

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
