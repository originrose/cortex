(ns scrabble.core
  (:require
    [clojure.core.matrix :as mat]
    [cortex.optimise :as opt]
    [cortex.network :as net]))

(mat/set-current-implementation :vectorz)

(def scrabble-values {\a 1 \b 3 \c 3 \d 2 \e 1 \f 4 \g 2 \h 4 \i 1 \j 8 \k 5
                      \l 1 \m 3 \n 1 \o 1 \p 3 \q 10 \r 1 \s 1 \t 1 \u 1 \v 4
                      \w 4 \x 8 \y 4 \z 10})

(defn run-experiment []
  (let [training-data   (mapv #(into [] [[(- (int %) (int \a))]]) (keys scrabble-values))
        training-labels (mapv #(into [] [[%]])                    (vals scrabble-values))
        network (net/sequential-network
                  [(net/linear-layer :n-inputs 1 :n-outputs 10)
                   (net/sigmoid-activation 10)
                   (net/linear-layer :n-inputs 10 :n-outputs 1)])
        n-epochs 10000
        learning-rate 0.3
        momentum 0.9
        batch-size 1
        loss-fn (opt/quadratic-loss)
        optimizer (net/sgd-optimizer network loss-fn learning-rate momentum)]
    (println "Training network...")
    (net/train-network optimizer n-epochs batch-size training-data training-labels)

    (println "Evaluating network...")
    (let [[results score] (net/evaluate network training-data training-labels)
          label-count (count training-data)
          score-percent (float (/ score label-count))]
      (println (format "Score: %f [%d of %d]" score-percent score label-count)))))

