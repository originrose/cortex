(ns cortex.suite.logistic-test
  (:require [clojure.java.io :as io]
            [clojure.string :as s]
            [clojure.test :refer :all]
            [cortex.dataset :as dataset]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
            [cortex.nn.traverse :as traverse]
            [cortex.suite.train :as train]
            [cortex.suite.inference :as inference]))

;; This test namespace uses the logistic classifier to learn to predict whether
;; people will default based on their balance and income.

;; "Default" dataset taken from: https://cran.r-project.org/web/packages/ISLR/index.html
;; Distributed under the GPL-2 LICENSE
;; The format is USER_ID,STUDENT?,DEFAULT?,BALANCE,INCOME

(defn default-dataset
  []
  (->> "default.csv"
      (io/resource)
      (slurp)
      (s/split-lines)
      (rest)                                     ;; ignore the header row
      (map (fn [l] (drop 2 (s/split l #"," ))))  ;; ignore id, student cols
      (mapv (fn [[default balance income]]
             {:data [(Double. balance) (Double. income)]
              :labels (if (= default "\"Yes\"") [1.0] [0.0])}))))

(def description
  [(layers/input 2)
   (layers/batch-normalization 0.9)
   (layers/linear->logistic 1)])

(deftest logistic-test
  (io/delete-file "trained-network.nippy" true)
  (let [default-data (default-dataset)
        inf-data (repeatedly #(rand-nth default-data))
        dataset (dataset/map-sequence->dataset inf-data (/ (count default-data) 20))
        network (-> description network/build-network traverse/auto-bind-io)
        _ (train/train-n dataset description network :batch-size 50 :epoch-count 300)
        trained-network (train/load-network "trained-network.nippy" description)
        [[should-def] [shouldnt-def]] (inference/infer-n-observations trained-network [[5000.0 10.0] [5.0 100000.0]]
                                                                      2 :batch-size 2)]
    (is (> should-def 0.99))
    (is (< shouldnt-def 0.01))))
