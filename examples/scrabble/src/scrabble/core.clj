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

(defn run-experiment []
  (let [;; convert the values from the scrabble pieces into 1x1 matrices
        training-data   (mapv #(into [] [[(- (int %) (int \a))]]) (keys scrabble-values))

        ;; convert the "labels" of the scrabble pieces into 1x1 matrices
        training-labels (mapv #(into [] [[%]])                    (vals scrabble-values))

        ;; This defines the shape of the network
        ;; TODO: add some explanation for the intution of the size of the various layers
        network (net/sequential-network
                  [(net/linear-layer :n-inputs 1 :n-outputs 10)
                   (net/sigmoid-activation 10)
                   (net/linear-layer :n-inputs 10 :n-outputs 1)])

        n-epochs 10000            ;; Number of iterations to train
        learning-rate 0.3
        momentum 0.9
        batch-size 1
        loss-fn (opt/quadratic-loss)
        optimizer (net/sgd-optimizer network loss-fn learning-rate momentum)]

    ;; Run the training portion of the algorithm using the parameters from above
    (println "Training network...")
    (net/train-network optimizer n-epochs batch-size training-data training-labels)

    ;; Once the network has been evaluated, use the same input data to test to see
    ;; how the neural network has learned the target function. We use the same
    ;; data for testing as training for this degenerate case.
    (println "Evaluating network...")
    (let [[results score] (net/evaluate network training-data training-labels)
          label-count (count training-data)
          score-percent (float (/ score label-count))]
      (println (format "Score: %f [%d of %d]" score-percent score label-count)))))

