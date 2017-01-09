(ns think.cortex.keras.core-test
  (:require [clojure.test :refer :all]
            [think.cortex.keras.core :as keras]
            [cortex.nn.network :as network]
            [clojure.core.matrix :as m]
            [cortex.verify.nn.import :as import]
            [think.compute.nn.compute-execute :as ce]))


(def simple_archf "models/simple_mnist.json")
(def simple_weightf "models/simple_mnist.h5")
(def simple_outf "models/simple_mnist_output.h5")


(deftest match-padding-correct
  "Verify that we get the correct padding value for same or valid padding."
  (is (= [2 2] (keras/match-padding {:border_mode "same" :nb_col 5 :nb_row 5})))
  (is (= [1 1] (keras/match-padding {:border_mode "same" :nb_col 3 :nb_row 3})))
  (is (= [3 3] (keras/match-padding {:border_mode "same" :nb_col 7 :nb_row 7})))
  (is (= [0 0] (keras/match-padding {:border_mode "valid" :nb_col 3 :nb_row 3})))
  (is (= [1 2] (keras/match-padding {:padding [1 2]}))))


(deftest keras-json-load
  "This test ensures that we get back a model we can load into a valid cortex
  description."
  (let [keras-model (keras/read-json-model simple_archf)
        model-desc  (keras/model->simple-description keras-model)]
    ;; these are known properties of simple model, if model changes,
    ;; update this part of test.
    (is (= "1.1.2" (:keras_version keras-model)))
    (is (= 12 (count (:config keras-model)) (count model-desc)))
    ;; within intersection of layer types both keras and cortex use,
    ;; the models should each contain those in the same order.
    (is (= (remove #(.contains ^String (str %) "flatten")
                   (map (comp keyword :name :config) (:config keras-model)))
           (rest (map :id model-desc))))))


(deftest network-builds
  "Ensure that the model we read in from Keras can actually be built, and
  that built result is correct."
  (let [keras-model (keras/read-json-model simple_archf)
        model-desc  (keras/model->simple-description keras-model)
        built-net   (network/build-network model-desc)]
    (is (= 1630602 (:parameter-count built-net)))
    (is (nil? (:verification-failures built-net)))))


(deftest read-outputs-correctly
  "Ensures that we read in output arrays for all layers that have them."
  (let [outputs  (keras/hdf5-layer-outputs simple_outf)
        out-arrs (for [[lyr arr] outputs] arr)]
    ;; all outputs are double arrays
    (is (every? #(instance? (Class/forName "[D") %) out-arrs))
    ;; just one spot check on dims of an output for now
    (is (= [12544] (m/shape (:convolution2d_2 outputs))))))


(deftest verify-simple-mnist
  "This is a basic model which has no ambiguity introduced by uneven strides,
  where frameworks start to differ. A failure here indicates a very basic
  problem in the Keras importer.

  Model does, however, inclue Dropout, so requires handling in inference or 
  training step so layer is not skipped."
  (is (keras/import-model simple_archf simple_weightf simple_outf)))
