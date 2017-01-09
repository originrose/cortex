(ns think.cortex.keras.core-test
  (:require [clojure.test :refer :all]
            [think.cortex.keras.core :as keras]
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


(deftest verify-simple-mnist
  "This is a basic model which has no ambiguity introduced by uneven strides,
  where frameworks start to differ. A failure here indicates a very basic
  problem in the Keras importer.

  Model does, however, inclue Dropout, so requires handling in inference or 
  trainin step so layer is not skipped."
  (is (keras/load-sidecar-and-verify simple_archf simple_weightf simple_outf)))
