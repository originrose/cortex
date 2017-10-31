# =============================== #
# Walkthrough of original example #
# =============================== #

This document is an interactive document. It is a walkthrough of the code
originally used for the mnist-classification example. The code can be
evaluated one at the time throughout the document. And you can change
whatever you want and evaluate that.

# Using this document
# ===================

To use this document as intended, it should be started using the command
"lein liq" in the examples/mnist-classification folder.

Use arrow keys to navigate, or keys "i", "j", "k", "l" (when the cursor is blue).

Use TAB to switch between navigation mode (blue cursor) and text mode (green
cursor).

Find more keybindings here: https://github.com/mogenslund/liquid/wiki/Cheat-Sheet

To evaluate an expression use "e" key in navigation mode. Praktice by evaluting
the sample expressions below (Type "1" to hightlight the s-expression that will
be valuted and type "1" again to remove the highlight.):

    (range 10) ; Move cursor inside parenthesis an click "e". Observe the output
               ; in the -prompt- window

    (p (map #(* % %) (range 10))) ; Try pressing "e" on "range" or on "p".

    (editor/end-of-line) ; This is an action doing something to the editor itself.

Notice, what is evaluated, depends on the position of the cursor. In some sense
it is the smallest complete s-expression containing the cursor.
So to load a function into memory press "e" while on "defn" or the name of the
function.

:WARNING: Read the instructions carefully, some commands in the document can take
          some time and need to finish, before continuing.


# 1. Global settings
# ==================

Evaluate the forms below to set settings that will be used for the rest of this
document:

    (def image-size 28)
    (def num-classes 10)
    
    (println image-size "," num-classes) ; Inspect


# 2. Download images and save to folder
# =====================================

This section will focus on downloading the training and test data (images) into
folders.


# Helper functions
# ----------------

Evaluate this two functions to load them into memory. The first one transforms
image date into png data and the second one generates a name and saves the
data to disk.

    (defn- ds-data->png
      "Given data from the original dataset, use think.image to produce png data."
      [ds-data]
      (let [data-bytes (byte-array (* image-size image-size))
            num-pixels (alength data-bytes)
            retval (image/new-image image/*default-image-impl*
                                    image-size image-size :gray)]
        (c-for [idx 0 (< idx num-pixels) (inc idx)]
               (let [[x y] [(mod idx image-size)
                            (quot idx image-size)]]
                 (aset data-bytes idx
                       (unchecked-byte (* 255.0
                                          (+ 0.5 (m/mget ds-data y x)))))))
        (image/array-> retval data-bytes)))


    (defn- save-image!
      "Save a dataset image to disk."
      [output-dir [idx {:keys [data label]}]]
      (let [image-path (format "%s/%s/%s.png" output-dir (util/max-index label) idx)]
        (when-not (.exists (io/file image-path))
          (io/make-parents image-path)
          (i/save (ds-data->png data) image-path))
        nil))


# Download data sets
# ------------------

:WARNING The two first form might take some seconds (maybe 10) to evaluate,
         do not evalute the next one until the first evaluation has
         finished. You can see the variable name in the -prompt- buffer
         to the left, when evaluation has finished.

The data is downloaded from
https://s3-us-west-2.amazonaws.com/thinktopic.datasets/mnist/

    (def training-dataset (mnist/training-dataset))
    (def test-dataset (mnist/test-dataset))

    (def dataset-folder "mnist/")

Evaluate this "defounce" form to load the function into memory.

    (defonce ensure-images-on-disk!
      (memoize
       (fn []
         (println "Ensuring image data is built, and available on disk.")
         (dorun (map (partial save-image! (str dataset-folder "training"))
                     (map-indexed vector training-dataset)))
         (dorun (map (partial save-image! (str dataset-folder "test"))
                     (map-indexed vector test-dataset)))
         :done)))

Evaluate this form to actually execute the function:

    (ensure-images-on-disk!)


# 3. Train the network
# ====================

Now data has been loaded and saved to disk. In this section the focus is
on training the network with the training data.

Evaluate the two functions to load them into memory, they will be
needed later:

    (defn- image-aug-pipeline
      "Uses think.image augmentation to vary training inputs."
      [image]
      (let [max-image-rotation-degrees 25]
        (-> image
            (image-aug/rotate (- (rand-int (* 2 max-image-rotation-degrees))
                                 max-image-rotation-degrees)
                              false)
            (image-aug/inject-noise (* 0.25 (rand))))))

    (defn- mnist-observation->image
      "Creates a BufferedImage suitable for web display from the raw data
      that the net expects."
      [observation]
      (patch/patch->image observation image-size))

The classification experiment system needs a way to go back and forth from
softmax indexes to string class names.

    (def class-mapping
      {:class-name->index (zipmap (map str (range 10)) (range))
       :index->class-name (zipmap (range) (map str (range 10)))})

Evalute "train-input" to load the input settings for the neural network:

    (def train-input
       {:options {:force-gpu? false
                  :live-updates? false
                  :tensorboard-output false}
        :arguments []
        :summary "Skipped"
        :errors nil})

The "initial-description" form is the definition of the neural network,
before any training has started. Multiple layers are defined with
different abilities:

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
       (layers/relu :center-loss {:label-indexes {:stream :labels}
                                  :label-inverse-counts {:stream :labels}
                                  :labels {:stream :labels}
                                  :alpha 0.9
                                  :lambda 1e-4})
       (layers/dropout 0.5)
       (layers/linear num-classes)
       (layers/softmax :id :labels)])

:WARNING The next form will run forever when evaluated. Because of that it
         has been encapsulted into a "future" to allow os to stop the
         execution, when we decide the network has been trained enough.

Evaluate the "trainer" form. While running a status can be found using a
webbrowser on this url: http://localhost:8091.

During traning a nippy file will be created and overwritten whenever a
better version of the network has been trained.

Watch the examples/mnist-classification folder to see when the nippy file
is created and updated.

Stop the training be evaluating the "future-cancel" command further down.

    (def trainer (future
       (with-out-str
       (let [training-folder (str dataset-folder "training")
             test-folder (str dataset-folder "test")
             train-ds (-> training-folder
                          (experiment-util/create-dataset-from-folder class-mapping
                                                                      :image-aug-fn (:image-aug-fn train-input))
                          (experiment-util/infinite-class-balanced-dataset))
             test-ds (-> test-folder (experiment-util/create-dataset-from-folder class-mapping))
             listener (if-let [file-path (:tensorboard-output train-input)]
                        (classification/create-tensorboard-listener {:file-path file-path})
                        (classification/create-listener
                          mnist-observation->image
                          class-mapping
                          train-input))]
         (classification/perform-experiment
           (initial-description image-size image-size num-classes)
           train-ds test-ds listener)))))

    (future-done? trainer)  ; See if the future is running
    (future-cancel trainer) ; Stop the training


# 4. Use network
# ==============

In this section we will load the network from a nippy file and apply it to
classify some images.

This is actually very simple:

 1. Load the network from a nippy file
 2. Apply the network to an image file to  guess what the written number is

Load the network file:

    (def network-filename (str train/default-network-filestem ".nippy"))

Load the file to evaluate. Change the path to evalute on some other file:

    (def f (io/file "mnist/test/4/2900.png"))

Load the data from the file and convert it to something the network is able to read:

    (def data1 (#'experiment-util/image->observation-data
                 (i/load-image f) :float :gray true identity))

The function experiment-util/image->observation-data is private, so "#'" was needed
to make it available anyway.

See the data:

    (println data1)

See the data as an image (You have to close the dialog with the image to proceed afterwards)

    (i/show (mnist-observation->image data1))

Use the network to guess the number:

    (util/max-index
      (:labels
        (first
          (execute/run
            (util/read-nippy-file network-filename)
            [{:data data1
              :labels [0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0]}]))))
