(ns thinktopic.cortex.network-test
  (:require
    [clojure.test :refer [deftest is are]]
    [thinktopic.cortex.network :as net]))


; a	b	| a XOR b
; 1	1	     0
; 0	1	     1
; 1	0	     1
; 0	0	     0
(def XOR-DATA [[1 1] [0 1] [1 0] [0 0]])
(def XOR-LABELS [[0] [1] [1] [0]])

(defn xor-test
  []
  (let [net (net/sequential-network
              [(net/linear-layer :n-inputs 2 :n-outputs 3)
               (net/sigmoid-activation 3)
               (net/linear-layer :n-inputs 3 :n-outputs 2)])
        optim-options {:loss (net/quadratic-loss)
                       :n-epochs 4000
                       :batch-size 1
                       :learning-rate 0.3}
        ;trained (sgd net optim-options XOR-DATA XOR-LABELS)
        ;[results score] (evaluate trained XOR-DATA XOR-LABELS)
        ;label-count (count XOR-LABELS)
        ;score-percent (float (/ score label-count))
        ]
    ;(println (format "XOR Score: %f [%d of %d]" score-percent score label-count))
    nil
    ))


;(defn hand-test
;  []
;  (let [net (network [2 3 1])
;        net (assoc net
;                   :biases [[0 0 0] [0]]
;                   :weights [[[1 1] [1 1] [1 1]]
;                             [[1 -2 1]]])
;        [results score] (evaluate net XOR-DATA XOR-LABELS)
;        label-count (count XOR-LABELS)
;        score-percent (float (/ score label-count))]
;    (println (format "XOR Score: %f [%d of %d]" score-percent score label-count))))


(deftest confusion-test
  (let [cf (net/confusion-matrix ["cat" "dog" "rabbit"])
        cf (-> cf
            (net/add-prediction "dog" "cat")
            (net/add-prediction "dog" "cat")
            (net/add-prediction "cat" "cat")
            (net/add-prediction "cat" "cat")
            (net/add-prediction "rabbit" "cat")
            (net/add-prediction "dog" "dog")
            (net/add-prediction "cat" "dog")
            (net/add-prediction "rabbit" "rabbit")
            (net/add-prediction "cat" "rabbit")
            )]
    (net/print-confusion-matrix cf)
    (is (= 2 (get-in cf ["cat" "dog"])))))
