(ns diabolo.scrabble
  (:use [nuroko.lab core charts])
  (:use [nuroko.gui visual])
  (:use [clojure.core.matrix])
  (:require [task.core :as task])
  (:import [mikera.vectorz Op Ops])
  (:import [mikera.vectorz.ops ScaledLogistic Logistic Tanh])
  (:import [nuroko.coders CharCoder])
  (:import [mikera.vectorz AVector Vectorz]))

;; some utility functions
(defn feature-img [vector]
  ((image-generator :width 28 :height 28 :colour-function weight-colour-mono) vector))

(def scores (sorted-map \a 1,  \b 3 , \c 3,  \d 2,  \e 1,
                        \f 4,  \g 2,  \h 4,  \i 1,  \j 8,
                        \k 5,  \l 1,  \m 3,  \n 1,  \o 1,
                        \p 3,  \q 10, \r 1,  \s 1,  \t 1,
                        \u 1,  \v 4,  \w 4,  \x 8,  \y 4,
                        \z 10))

(def score-coder (int-coder :bits 4))
(encode score-coder 3)
(decode score-coder *1)

(def letter-coder
  (class-coder :values (keys scores)))
(encode letter-coder \c)

(def task
  (mapping-task scores
                :input-coder letter-coder
                :output-coder score-coder))

(def net
  (neural-network :inputs 26
                  :outputs 4
                  :hidden-op Ops/LOGISTIC
                  :output-op Ops/LOGISTIC
                  :hidden-sizes [6]))

(show (network-graph net :line-width 2)
      :title "Neural Net : Scrabble")

(defn scrabble-score [net letter]
  (->> letter
       (encode letter-coder)
       (think net)
       (decode score-coder)))

(scrabble-score net \a)


;; evaluation function
(defn evaluate-scores [net]
  (let [net (.clone net)
        chars (keys scores)]
    (count (for [c chars
                 :when (= (scrabble-score net c) (scores c))] c))))

(show (time-chart
        [#(evaluate-scores net)]
        :y-max 26)
      :title "Correct letters")

;; training algorithm
(def trainer (supervised-trainer net task :batch-size 100))

(task/run
  {:sleep 1 :repeat 1000} ;; sleep used to slow it down, otherwise trains instantly.....
  (trainer net))

(scrabble-score net \x)
