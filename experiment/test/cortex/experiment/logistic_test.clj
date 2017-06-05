(ns cortex.experiment.logistic-test
  (:require [clojure.java.io :as io]
            [clojure.string :as s]
            [clojure.test :refer :all]
            [cortex.nn.layers :as layers]
            [cortex.nn.execute :as execute]
            [cortex.experiment.train :as train]
            [cortex.nn.network :as network]))

;; This test namespace uses the logistic classifier to learn to predict whether
;; people will default based on their balance and income.

;; "Default" dataset taken from: https://cran.r-project.org/web/packages/ISLR/index.html
;; Distributed under the GPL-2 LICENSE
;; The format is USER_ID,STUDENT?,DEFAULT?,BALANCE,INCOME

(defn default-dataset
  []
  (->> "test/data/default.csv"
      (slurp)
      (s/split-lines)
      (rest)                                     ;; ignore the header row
      (map (fn [l] (drop 2 (s/split l #"," ))))  ;; ignore id, student cols
      (mapv (fn [[^String default ^String balance ^String income]]
             {:data [(Double. balance) (Double. income)]
              :labels (if (= default "\"Yes\"") [1.0] [0.0])}))))

(def description
  [(layers/input 2 1 1 :id :data)
   (layers/batch-normalization)
   ;;Fix the weights to make the unit test work.
   (layers/linear 1 :weights [[-0.2 0.2]])
   (layers/logistic :id :labels)])

(deftest logistic-test
  (io/delete-file "trained-network.nippy" true)
  (let [ds (shuffle (default-dataset))
        ds-count (count ds)
        train-ds (take (int (* 0.9 ds-count)) ds)
        test-ds (drop (int (* 0.9 ds-count)) ds)
        _ (train/train-n description train-ds test-ds
                         :batch-size 50 :epoch-count 10
                         :simple-loss-print? true)
        trained-network (train/load-network "trained-network.nippy")
        input-data [{:data [5000.0 10.0]} {:data [5.0 100000.0]}]
        [[should-def] [shouldnt-def]] (->> (execute/run trained-network input-data)
                                           (map :labels))]
    (is (> should-def 0.96))
    (is (< shouldnt-def 0.02))))
