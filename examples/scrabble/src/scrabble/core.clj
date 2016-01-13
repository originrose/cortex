(ns scrabble.core
  (:require
    [clojure.core.matrix :as m]
    [cortex.optimise :as opt]
    [cortex.layers :as layers]
    [cortex.optimise :as opt]
    [cortex.core :as core]
    [clojure.core.matrix.random :as rand]
    [cortex.util :as util]
    [cortex.protocols :as cp]
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

(defn num->bit-array [input & {:keys [bits]
                               :or {bits 4}}]
  "Takes a number, returns a vector of floats representing the number in
  binary. Optionally specify the number of bits for padding."
 (->> (clojure.pprint/cl-format nil (str "~" bits ",'0',B") input)
      (mapv #(new Double (str %)))))

(defn char->bit-array [character]
  "Takes a character, returns and bit vector with a 1.0 in the nth position
  where n is the offset from the character 'a'."
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
(def n-epochs 1) ;;epoch is a pass through the dataset
(def learning-rate 0.1)
(def momentum 0.1)
(def batch-size 1) ;;For this example a small batch size is ideal
(def loss-fn (opt/mse-loss))

(defn random-matrix
  [shape-vector]
  (if (> (count shape-vector) 1 )
    (apply util/weight-matrix shape-vector)
    (rand/sample-normal (first shape-vector))))

(defn linear-layer
  [n-inputs n-outputs]
  (layers/linear (random-matrix [n-outputs n-inputs])
                 (random-matrix [n-outputs])))

;; training, evaluation and prediction
(defn create-network
  []
  (let [network-modules [(linear-layer input-width hidden-layer-size)
                         (layers/logistic [hidden-layer-size])
                         (linear-layer hidden-layer-size output-width)]]
    (core/stack-module network-modules)))

(defn create-optimizer
  [network]
  ;(opt/adadelta-optimiser (core/parameter-count network))
  (opt/sgd-optimiser (core/parameter-count network) {:learn-rate learning-rate :momentum momentum} )
  )

(defn train-step
  [input answer network loss-fn]
  (let [network (core/forward network input)
        temp-answer (core/output network)
        loss (cp/loss loss-fn temp-answer answer)
        loss-gradient (cp/loss-gradient loss-fn temp-answer answer)]
    (core/backward (assoc network :loss loss) input loss-gradient)))

(defn test-train-step
  []
  (train-step (first training-data) (first training-labels) (create-network) loss-fn))

(defn train-batch
  [input-seq label-seq network optimizer loss-fn]
  (let [network (reduce (fn [network [input answer]]
                          (train-step input answer network loss-fn))
                        network
                        (map vector input-seq label-seq))]
    (core/optimise optimizer network)))

(defn train
  []
  (let [network (create-network)
        optimizer (create-optimizer network)
        epoch-batches (repeatedly n-epochs
                                  #(into [] (partition batch-size (shuffle (range (count training-data))))))
        epoch-count (atom 0)
        [optimizer network] (reduce (fn [opt-network batch-index-seq]
                                      (swap! epoch-count inc)
                                      (println "Running epoch:" @epoch-count)
                                      (reduce (fn [[optimizer network] batch-indexes]
                                                (let [input-seq (mapv training-data batch-indexes)
                                                      answer-seq (mapv training-labels batch-indexes)
                                                      [optimizer network] (train-batch input-seq
                                                                                       answer-seq
                                                                                       network optimizer loss-fn)]
                                                  ;(println "loss after batch:" (:loss network))
                                                  [optimizer network]))
                                              opt-network
                                              batch-index-seq))
                                    [optimizer network]
                                    epoch-batches)]
    network))


(defn evaluate
  [network]
  (let [test-results (map (fn [input]
                            (let [network (core/forward network input)]
                              (m/emap #(Math/round (double %)) (core/output network))))
                          training-data)
        correct (count (filter #(m/equals (first %) (second %)) (map vector test-results training-labels)))]
    (double (/ correct (count training-data)))))



(defn train-and-evaluate
  []
  (let [network (train)
        fraction-correct (evaluate network)]
    (println (format "Network score: %g" fraction-correct))))

(defn -main
  [& args]
  (train-and-evaluate))
