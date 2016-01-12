(ns scrabble.core
  (:require
    [clojure.core.matrix :as mat]
    [cortex.optimise :as opt]
    [cortex.network :as net]))

;; This is an example of a very simple network that is designed to learn the
;; values for given scrabble pieces. Inspired by the following talk:
;;  - "Machine Learning Live" by Mike Anderson (see: https://www.youtube.com/watch?v=QJ1qgCr09j8)
;;
;; It is a bit of a degenerate case as the "test" data is the same as the
;; training data, however it shows how a neural network can learn an arbitrary
;; function.

(mat/set-current-implementation :vectorz)

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

;; training, evaluation and prediction

(defn train [network training-data training-labels]
  (let [n-epochs 30            ;; Number of iterations to train
        learning-rate 0.3
        momentum 0.1
        batch-size 1
        loss-fn (opt/quadratic-loss)
        optimizer (net/sgd-optimizer network loss-fn learning-rate momentum)]
    (net/train-network optimizer n-epochs batch-size training-data training-labels)))

(defn evaluate [network test-data test-labels]
  (let [[results score] (net/evaluate network test-data test-labels)
        label-count (count test-data)
        score-percent (float (/ score label-count))]
    (println (format "Score: %f [%d of %d]" score-percent score label-count))))

(defn predict [network character]
  (-> (net/predict network [(char->bit-array character)])
      first
      bit-array->num))

(defn run-experiment []
  (let [;; convert the values from the scrabble pieces (letters) into floating-point bit-arrays
        training-data (for [k (keys scrabble-values)] [(char->bit-array k)])

        ;; convert the "labels" of the scrabble pieces (scores) into floating-point bit-arrays
        training-labels (for [v (vals scrabble-values)] [(num->bit-array v)])

        ;; infer the width of the input and output based on the shape of the training data
        ;; - input layer has one input corresponding to each possible letter
        ;; - output layer is of size 4 as we can represent the corresponding scores for
        ;;   each letter with 4 bits (max score of 10)
        input-width (last (mat/shape training-data))
        output-width (last (mat/shape training-labels))

        ;; This defines the shape of the network
        ;; TODO: add some explanation for the intuition of the size of the hidden layer
        network (net/sequential-network
                  [(net/linear-layer :n-inputs input-width :n-outputs 6)
                   (net/sigmoid-activation 6)
                   (net/linear-layer :n-inputs 6 :n-outputs output-width)])]

    ;; Run the training portion of the algorithm using the parameters from above
    (train network training-data training-labels)

    ;; Once the network has been trained, use the same input data to test to see
    ;; how the neural network has learned the target function. We use the same
    ;; data for testing as training for this degenerate case.
    (evaluate network training-data training-labels)

    ;; Use the network to perform a prediction
    (println (format "Piece 'k' is worth %d points." (predict network \k)))))
