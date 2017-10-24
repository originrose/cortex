# resnet-retrain

[dataset](http://vision.ucmerced.edu/datasets/landuse.html)


unzip to data directory.  Should have this directory structure:

```
data/UCMerced_LandUse/Images
```

Run resize_images.sh
 - (you may need to install imagemagick `brew install imagemagick`)
 - you may also need to run with sudo/and or fix the permissions depending on your system setup
 - at the end you want something to look like ` ls -l  data/UCMerced_LandUse/Images/buildings/buildings00.png ` which you can open and verify that the size is 224x224


Get the RESNET 50 model
./get-resnet50.sh 

- you should now have a directory `/models` with a resnet50.nippy file in it

Start repl

```
(create-train-test-folders "data/UCMerced_LandUse/Images")
```
- This will setup the training pictures into the correct training and test folder structure that cortex expects under the `data` directory.

You should be ready to train now.


```
(train)
```

Or if you have mem issues, you might want to try it from the uberjar

`lein uberjar` and then `java -jar target/resnet-retrain-0.9.23-SNAPSHOT.jar`

After only retaining the RESNET50 for only one epoch we get pretty good results. Try it out with

```
(label-one)
```


If you are interested in continuing the training, you can run train-again or from the uberjar `java -jar target/resnet-retrain.jar batch-size true`


Copyright © 2017 FIXME

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
