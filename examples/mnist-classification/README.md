# mnist-classification

This example uses the `experiment/classification` machinery to exemplify continual training with a convolutional neural network, visualization of that process, as well as further inference of images from the dataset. It uses the classic MNIST handwritten digit corpus. The training dataset also employs augmentation to slightly change each training image, creating variations on the classic inputs.

## Usage

Cortex training supports gpu (CUDA) and cpu training. CUDA setup is explained in the `cortex` README.

Notably, if you're using an older 7.5 version of CUDA you will need to adjust the `[org.bytedeco.javacpp-presets/cuda "7.5-1.2"]` dependency in `project.clj`

You might need to install local snapshots versions of `cortex` and `experiment`. That is, run `lein install` in the `cortex` and `experiment` folders.

Then, `lein run`:

```
$ lein run
Welcome! Please wait while we compile some Clojure code...
Loading mnist training dataset.
Done loading mnist training dataset in 14.143s
Loading mnist test dataset.
Done loading mnist test dataset in 2.329s
Training forever from uberjar.
Ensuring image data is built, and available on disk.
Training forever.
Building dataset from folder: mnist/training
Building dataset from folder: mnist/test
Gate opened on http://localhost:8091
Training network:

|                 type |            input |           output |  :bias | :centers | :means | :scale | :variances |   :weights |
|----------------------+------------------+------------------+--------+----------+--------+--------+------------+------------|
|       :convolutional |    1x28x28 - 784 | 20x24x24 - 11520 |   [20] |          |        |        |            |    [20 25] |
|         :max-pooling | 20x24x24 - 11520 |  20x12x12 - 2880 |        |          |        |        |            |            |
|             :dropout |  20x12x12 - 2880 |  20x12x12 - 2880 |        |          |        |        |            |            |
|                :relu |  20x12x12 - 2880 |  20x12x12 - 2880 |        |          |        |        |            |            |
|       :convolutional |  20x12x12 - 2880 |    50x8x8 - 3200 |   [50] |          |        |        |            |   [50 500] |
|         :max-pooling |    50x8x8 - 3200 |     50x4x4 - 800 |        |          |        |        |            |            |
| :batch-normalization |     50x4x4 - 800 |     50x4x4 - 800 |  [800] |          |  [800] |  [800] |      [800] |            |
|              :linear |     50x4x4 - 800 |  1x1x1000 - 1000 | [1000] |          |        |        |            | [1000 800] |
|                :relu |  1x1x1000 - 1000 |  1x1x1000 - 1000 |        |          |        |        |            |            |
|             :dropout |  1x1x1000 - 1000 |  1x1x1000 - 1000 |        |          |        |        |            |            |
|              :linear |  1x1x1000 - 1000 |      1x1x10 - 10 |   [10] |          |        |        |            |  [10 1000] |
|             :softmax |      1x1x10 - 10 |      1x1x10 - 10 |        |          |        |        |            |            |
Parameter count: 849780
Classification accuracy: 0.9018
Saving network to trained-network.nippy
Classification accuracy: 0.9209
Saving network to trained-network.nippy
Classification accuracy: 0.9345
Saving network to trained-network.nippy
Classification accuracy: 0.9405
Saving network to trained-network.nippy
Classification accuracy: 0.9467
Saving network to trained-network.nippy
Classification accuracy: 0.9576
Saving network to trained-network.nippy
Classification accuracy: 0.9556
Classification accuracy: 0.9587
Saving network to trained-network.nippy
Classification accuracy: 0.9559
Classification accuracy: 0.9668
Saving network to trained-network.nippy
```

You will see output about compilation and building the image sets. The program also starts a web server at [http://localhost:8091/](http://localhost:8091/) that hosts a page that updates live with results and information about the training process, including a confusion matrix visualization. After you see `Gate opened on http://localhost:8091`, go ahead and open a web page.

After each training epoch the new network's classification accuracy is evaluated against a test dataset. If the new network is found to be the best so far (on that metric), then the trained model is saved to the file: `trained-network.nippy`.

When you think it has run long enough, go ahead and stop the process.

Get a new REPL up in the core namespace and run the `label-one` function:

```
mnist-classification.core> (label-one)
Ensuring image data is built, and available on disk.
Building dataset from folder: mnist/test
{:answer 2, :guess 2}
```

A randomly selected handwritten digit from the test dataset is chosen, shown in a popup, and classified using the last exported network. Notably, the `label-one` function only uses functionality from base `cortex`, and none from the `experiment` framework. `cortex` as a library has many fewer dependencies, which simplifies deployment of systems that only need inference without the complexities of the experiment training machinery.

## Running from uberjar

You can also build as a uberjar and run the entire process outside of Lein. It starts a lot faster, runs a bit faster, and hopefully hints at a more repeatable process.

`lein uberjar`

and then `java -jar target/classify-example.jar`

Copyright © 2016 ThinkTopic, LLC.
