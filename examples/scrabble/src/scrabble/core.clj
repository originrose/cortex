(ns scrabble.core
  (:require
    [clojure.core.matrix :as m]
    [cortex.nn.layers :as layers]
    [cortex.optimise :as opt]
    [cortex.nn.core :as core]
    [cortex.nn.network :as net]
    [clojure.pprint])
  (:gen-class))

;; This is an example of a very simple network that is designed to learn the
;; values for given scrabble pieces. Inspired by the following talk:
;;  - "Machine Learning Live" by Mike Anderson (see: https://www.youtube.com/watch?v=QJ1qgCr09j8)
;;
;; It is a bit of a degenerate case as the "test" data is the same as the
;; training data, however it shows how a neural network can learn an arbitrary
;; function.

(m/set-current-implementation :vectorz)

;; scrabble pieces and their values
(def scrabble-values {\a 1 \b 3 \c 3 \d 2 \e 1 \f 4 \g 2 \h 4 \i 1 \j 8 \k 5
                      \l 1 \m 3 \n 1 \o 1 \p 3 \q 10 \r 1 \s 1 \t 1 \u 1 \v 4
                      \w 4 \x 8 \y 4 \z 10})

;; coders

(defn num->bit-array
  "Takes a number, returns a vector of floats representing the number in
  binary. Optionally specify the number of bits for padding."
  [input & {:keys [bits] :or {bits 4}}]
  (->> (clojure.pprint/cl-format nil (str "~" bits ",'0',B") input)
       (mapv #(new Double (str %)))))

(defn char->bit-array
  "Takes a character, returns and bit vector with a 1.0 in the nth position
  where n is the offset from the character 'a'."
  [character]
  (let [offset (- (int character) (int \a))
        begin (take offset (repeatedly #(new Double 0.0)))
        end (take (- 26 offset 1) (repeatedly #(new Double 0.0)))]
    (into [] (concat begin [1.0] end))))

;; decoder

(defn bit-array->num [input]
  "Inverse of encoder, takes a bit array and converts it into an int"
  (-> (map int input)
       clojure.string/join
       (Integer/parseInt 2)))


(def training-data (into [] (for [k (keys scrabble-values)] (char->bit-array k))))

;; convert the "labels" of the scrabble pieces (scores) into floating-point bit-arrays
(def training-labels (into [] (for [v (vals scrabble-values)] (num->bit-array v))))

;; infer the width of the input and output based on the shape of the training data
;; - input layer has one input corresponding to each possible letter
;; - output layer is of size 4 as we can represent the corresponding scores for
;;   each letter with 4 bits (max score of 10)
(def input-width (last (m/shape training-data)))
(def output-width (last (m/shape training-labels)))
(def hidden-layer-size 4)
(def n-epochs 100) ;;epoch is a pass through the dataset
(def learning-rate 0.1)
(def momentum 0.1)
(def batch-size 1) ;;For this example a small batch size is ideal
(def loss-fn (opt/mse-loss))

;; training, evaluation and prediction
(defn create-network
  []
  (let [network-modules [(layers/linear-layer input-width hidden-layer-size)
                         (layers/logistic [hidden-layer-size])
                         (layers/linear-layer hidden-layer-size output-width)]]
    (core/stack-module network-modules)))

(defn create-optimizer
  [network]
  ;(opt/adadelta-optimiser (core/parameter-count network))
  (opt/sgd-optimiser learning-rate momentum)
  )



(defn train
  []
  (let [network (create-network)
        optimizer (create-optimizer network)]
    (net/train network optimizer loss-fn training-data training-labels batch-size n-epochs)))

(defn classify [network character]
  (let [score (-> (->> (net/run network [(char->bit-array character)])
                        (first)
                        (map #(if (> % 0.5) 1 0))
                        (clojure.string/join))
                  (Integer/parseInt 2))]
    (println character " is " score)))

(defn evaluate
  [network]
  (do
    (classify network \j)
    (classify network \k)
    (classify network \a)
    (classify network \q)
    (net/evaluate network training-data training-labels)))


(defn train-and-evaluate
  []
  (let [network (train)
        fraction-correct (evaluate network)]
    (println (format "Network score: %g" fraction-correct))))

(defn -main
  [& args]
  (train-and-evaluate))
