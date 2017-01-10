(ns cortex.dataset-test
  (:require [cortex.dataset :as ds]
            [clojure.test :refer :all]))


(deftest in-memory-dataset
  (let [test-ds (ds/create-in-memory-dataset {:data {:data (vec (repeat 1000 [1 2 3 4 5]))
                                                     :shape 5}
                                              :label {:data (vec (repeat 1000 [1 2]))
                                                      :shape 2}}
                                             {:training (vec (shuffle (range 500)))
                                              :cross-validation (vec (range 500 1000))
                                              :holdout (vec (range 500 1000))})
        epoch (ds/get-batches test-ds 10 :training [:data :data :label])]
    (is (= 50
           (count epoch)))
    (is (= {:data (vec (repeat 10 [1 2 3 4 5]))
            :label (vec (repeat 10 [1 2]))}
           (first epoch)))
    ;;this function has to correctly interpret the dataset so it is necessary that it
    (let [column-groups (vec (ds/batch-sequence->column-groups test-ds 10 :training [[:data :label] [:data] [:label]]))]
      (is (= 3 (count column-groups)))
      (is (= #{:data :label}
             (set (keys (nth column-groups 0)))))
      (is (= #{:data}
             (set (keys (nth column-groups 1)))))
      (is (= #{:label}
             (set (keys (nth column-groups 2)))))
      (is (= [1 2 3 4 5]
             (first (get-in column-groups [0 :data]))))
      (is (= [1 2]
             (first (get-in column-groups [2 :label])))))))


(deftest inifinite-dataset
  (let [data-epoch-seq (repeat (repeat [[1 2 3 4 5] [1 2]]))
        test-ds (ds/create-infinite-dataset [[:data 5] [:label 2]]
                                            data-epoch-seq
                                            data-epoch-seq
                                            data-epoch-seq)
        epoch (take 50 (ds/get-batches test-ds 10 :training [:data :data :label]))]
    (is (= 50
           (count epoch)))
    (is (= {:data (vec (repeat 10 [1 2 3 4 5]))
            :label (vec (repeat 10 [1 2]))}
           (first epoch)))

    (let [column-data (ds/batches->columns epoch)]
      (is (= 2 (count (keys column-data))))
      (is (= [1 2 3 4 5]
             (first (get-in column-data [:data]))))
      (is (= [1 2]
             (first (get-in column-data [:label])))))))




(deftest infinite-sequence->infinite-dataset
  (let [sequence (->> (range)
                      (map (fn [idx]
                             {:a (repeat 10 idx)
                              :b (repeat 10 [(rem (long idx) 5) (rem (long idx) 4) (rem (long idx) 3)])
                              :c (repeat 2 (quot (long idx) 5))})))
        test-ds (ds/map-sequence->dataset sequence 200)
        cv-item (first (ds/get-batches test-ds 5 :cross-validation [:a :b :c]))
        holdout-item (first (ds/get-batches test-ds 5 :holdout [:a :b :c]))
        training-item (first (ds/get-batches test-ds 5 :holdout [:a :b :c]))]
    (is (= {:a {:shape 10}, :b {:shape 30}, :c {:shape 2}}
           (ds/shapes test-ds)))
    (is (= [0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 4 4 4 4 4 4 4 4 4 4]
           (vec (flatten (get cv-item :a)))))))
