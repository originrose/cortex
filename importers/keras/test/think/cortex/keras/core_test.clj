(ns think.cortex.keras.core-test
  (:require [clojure.test :refer :all]
            [think.cortex.keras.core :as keras]
            [cortex.nn.network :as network]
            [clojure.core.matrix :as m]
            [cortex.verify.nn.import :as import]
            [cortex.graph :as graph]))


(def simple_archf "models/simple_mnist.json")
(def simple_weightf "models/simple_mnist.h5")
(def simple_outf "models/simple_mnist_output.h5")

(def resnet_archf "models/resnet50.json")
(def resnet_weightf "models/resnet50.h5")
(def resnet_outf "models/resnet50_output.h5")

(deftest match-padding-correct
  "Verify that we get the correct padding value for same or valid padding."
  (is (= [1 1] (keras/match-padding {:padding "same" :kernel_size [3 3]})))
  (is (= [3 3] (keras/match-padding {:padding "same" :kernel_size [7 7]})))
  (is (= [0 0] (keras/match-padding {:padding "valid" :kernel_size [3 3]})))
  (is (= [2 2] (keras/match-padding {:padding [2 2] :kernel_size [5 5]})))
  (is (= [3 3] (keras/match-padding {:padding [[3 3] [3 3]] :kernel_size [5 5]})))
  (is (thrown? Exception (keras/match-padding {:padding [[3 2] [3 5]] :kernel_size [5 5]}))))


(deftest mnist-keras-json-load
  "This test ensures that we get back a model we can load into a valid cortex
  description."
  (let [keras-model (keras/read-json-model simple_archf)
        model-desc  (keras/keras-json->cortex-desc simple_archf)]
    ;; these are known properties of simple model, if model changes,
    ;; update this part of test.
    (is (= "2.0.6" (:keras_version keras-model)))
    (is (= 11 (count (:config keras-model)) (count model-desc)))
    ;; within intersection of layer types both keras and cortex use,
    ;; the models should each contain those in the same order.
    (is (= (remove #(.contains ^String (str %) "flatten")
                   (map (comp keyword :name :config) (:config keras-model)))
           (rest (map :id model-desc))))))


(deftest mnist-network-builds
  "Ensure that the model we read in from Keras can actually be built, and
  that built result is correct."
  (let [model-desc (keras/keras-json->cortex-desc simple_archf)
        built-net (network/linear-network model-desc)]
    (is (= 422154 (graph/parameter-count (network/network->graph built-net))))))


(deftest mnist-read-outputs-correctly
  "Ensures that we read in output arrays for all layers that have them."
  (let [outputs  (keras/network-output-file->layer-outputs simple_outf)
        out-arrs (for [[lyr arr] outputs] arr)]
    ;; all outputs are double arrays
    (is (every? #(instance? (Class/forName "[D") %) out-arrs))
    ;; just one spot check on dims of an output for now
    (is (= [12544] (m/shape (:conv2d_2 outputs))))))


(deftest verify-simple-mnist
  "This is a basic model which has no ambiguity introduced by uneven strides,
  where frameworks start to differ. A failure here indicates a very basic
  problem in the Keras importer."
  (is (keras/import-model simple_archf simple_weightf simple_outf)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; ResNet50 testing

(deftest resnet-json-load
  "This test ensures that we get back a model we can load into a valid cortex
  description."
  (let [keras-model (keras/read-json-model resnet_archf)
        model-desc  (keras/keras-json->cortex-desc resnet_archf)]
    ;; these are known properties of resnet model, if model changes,
    ;; update this part of test.
    (is (= "2.0.6" (:keras_version keras-model)))
    (is (= 192 (count model-desc)))

    ;; overlapping layers in keras and cortex should match,
    ;; in the same order
    (is (= (remove #(or (.contains ^String (str %) "flatten")
                        (.contains ^String (str %) "input")
                        (.contains ^String (str %) "add"))
                   (map #(keyword (:name %)) (get-in keras-model [:config :layers])))
           (->> (rest model-desc) ;; drop first nil (input)
                (remove #(or (= (:type %) :split) ;; filter out splits and joins
                             (= (:type %) :join)
                             ;; filter out last activation layer (own layer in cortex, not in keras)
                             (clojure.string/ends-with? (name (:id %)) "activation")
                             ))
                (map :id))))

    ;; check that for each "Add", there is one split and one join
    (is (= 16
           (count (filter #(= (:class_name %) "Add") (get-in keras-model [:config :layers])))
           (count (filter #(= (:type %) :split) model-desc))
           (count (filter #(= (:type %) :join) model-desc))))))


(deftest resnet-builds
  "Test if we can build the right network from the resnet Cortex description."
  (let [model-desc (keras/keras-json->cortex-desc resnet_archf)
        resnet (network/linear-network model-desc)]
    (is (= 67253864 (graph/parameter-count (network/network->graph resnet))))))



(deftest ^:skip-ci verify-resnet
  (is (keras/import-model resnet_archf resnet_weightf resnet_outf)))
