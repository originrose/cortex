Cats and Dogs Classification using Cortex

![](resources/cat_and_dog_split.jpg)

# introduction

This was triggered and inspired by the post below:
http://gigasquidsoftware.com/blog/2016/12/27/deep-learning-in-clojure-with-cortex/

While the blog post is still very actual, the cortex code had been updated a bit, so this is an updated version of that classification example.

The goal of this exercice is to train a cortex network for classification, a network that can then differenciate between pictures of cats and dogs. 

The network will be trained with a large set of images from kaggle.com, and we will then see how to use the trained network.

The summary of the experiment is as below:

- download the images used to train the network (outside clojure)
- convert the images, and prepare a folder structure with them to be used in datasources for the next step ([prepare.clj](src/catsdogs/prepare.clj))
- train the network ([training.clj](src/catsdogs/training.clj))
- use remote images to validate the trained network ([simple](src/catsdogs/simple.clj))

# image (data) download

Download test.zip and train.zip from the following location:
https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data

Fill in the form, extract and put the content (all the pictures) inside a folder named: data-cats-dogs/original in the project directory.

# prepare

The preparation phase consist of creating images 

the image will be resized and converted to grayscale before the training.
The picture set is composed of only cats and dogs, with the type of the animal being in the original file name.

With a first half of the original picture set, we will constitute a training folder, 
where the picture will be used tell the network to remember that this picture a cat, or a dog explicitly.

With the second half of the original picture set, we will create a testing folder. 
Those pictures will be used to validate answers of the trained network with pictures. 

The main function is the preprocess-image function, that turn one of the original picture to a smaller size version in grayscale:

![resources/cat_training.png](TODO: Add working link to picture)

-  [idx [file label] ], unique index of the picture, source file, and a cat or dog label
-  output-dir, the top target folder
-  image-size, the new size of the target image

```
(defn preprocess-image
  "scale to image-size and convert the picture to grayscale"
  [output-dir image-size [idx [file label]]]
  (let [img-path (str output-dir "/" label "/" idx ".png" )]
    (when-not (.exists (io/file img-path))
      (io/make-parents img-path)
      (-> (imagez/load-image file)
          ((filters/grayscale))
          (imagez/resize image-size image-size)
          (imagez/save img-path)))))
```

The result of the prepare phase is to take all the pictures from the original folder,  and convert them to the given output. (given size, and grayscale). This gives the structure below:

```
.
├── original
├── testing
│   ├── cat
│   └── dog
└── training
      ├── cat
      └── dog
```

Once the preprocessing of images has been done, we can now go to the training network phase.

# training

For the training phase, we will setup two cortex datasources from the folders that were prepared in the previous phase.
- training
- testing

Setting up a datasource is done with the experiment-util/create-dataset-from-folder function:

```
(require '[cortex.experiment.util :as experiment-util])
(def train-ds
 (-> training-folder
   (experiment-util/create-dataset-from-folder
     class-mapping :image-aug-fn (:image-aug-fn {}))
   (experiment-util/infinite-class-balanced-dataset)))
```

The initial description of the network has been kept identical as the mnist classification example (although you might want to experiment with different configurations for example, taking the dropout layers out)

```
(defn initial-description
 [input-w input-h num-classes]
 [(layers/input input-w input-h 1 :id :data)
  (layers/convolutional 5 0 1 20)
  (layers/max-pooling 2 0 2)
  (layers/dropout 0.9)
  (layers/relu)
  (layers/convolutional 5 0 1 50)
  (layers/max-pooling 2 0 2)
  (layers/batch-normalization)
  (layers/linear 1000)
  (layers/relu :center-loss 
   {:label-indexes {:stream :labels}
    :label-inverse-counts {:stream :labels}
    :labels {:stream :labels}
    :alpha 0.9
    :lambda 1e-4})
  (layers/dropout 0.5)
  (layers/linear num-classes)
  (layers/softmax :id :labels)])
```

Finally, the network can be trained using another function from the cortex.experiement.classification namespace. Parameters are:
- the description of the network
- the training datasource
- the testing datasource
- a function to convert picture to show on the simple webapp (started when the experiment is running)
- a class-mapping (to go back and forth between the categories in alphabet, and in digits)
- options ... (none here)

```
(require '[cortex.experiment.classification :as classification])
(classification/perform-experiment
    (initial-description image-size image-size num-classes)
    train-ds
    test-ds
    observation->image
    class-mapping
    {})
```


While the experiment is running, a small web application is started to confirm the status of the network training:

[http://localhost:8091](http://localhost:8091)

![resources/web.png](TODO: Add working link to picture)

On each training iteration the network will be checked against images from the testing folder, and will be given a score. If that score is higher than the previous iteration, then the updated network will be saved in the nippy file.

# use the network

The trained network is periodically saved to a file with a default name of trained-network.nippy.
The nippy file is mostlly only a binary version of a big clojure map. 
First we load the network saved in a nippy  file:

```
(def nippy
  (util/read-nippy-file "trained-network.nippy"))
```

Then the main client function in simple.clj, is the guess function. It takes a loaded trained network, and an image-path and run the network with the converted input.

```
(defn guess [nippy image-path]
  (let [obs (image-file->observation image-path)]
    (-> (execute/run nippy [obs])
        first
        :labels
        util/max-index
        index->class-name)))
```

The steps are explained below:

1. convert the image to a compatible format (same size as network) and turn it into an observation, something that the network can act upon
2. make a run of the network. the input is an array of observations
3. the return object is also an array, one result for each element of the input array. (so take the first element)
4. basically, each result is a score for each of the possible input, so util/max-index gets one of the possible categories ("cat", "dog") and retrieves the one with the highest score

![resources/4-ways-cheer-up-depressed-cat.jpg](TODO: Add working link to picture)

```
catsdogs.simple=> (guess nippy "resources/4-ways-cheer-up-depressed-cat.jpg")
; CUDA backend creation failed, reverting to CPU
; "cat"
```
