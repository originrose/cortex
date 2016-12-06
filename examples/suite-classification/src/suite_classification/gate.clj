(ns suite-classification.gate
  (:require [think.gate.core :as gate]
            [suite-classification.core :as core]
            [cortex.suite.classification :as classification]
            [clojure.core.matrix :as m]))

(defn on-foo
  [params]
  (+ (:a params) 41))

(defn on-bar
  [params]
  {:a :b})


(defonce confusion-matrix (atom {}))


(defn reset-confusion-matrix
  [network-eval]
  (reset! confusion-matrix (classification/network-eval->rich-confusion-matrix network-eval))
  nil)


(defn get-confusion-matrix
  [& args]
  (let [current-matrix @confusion-matrix
        class-names (vec (sort (keys current-matrix)))]
    {:class-names class-names
     :matrix (mapv (fn [row-name]
                     (mapv (fn [col-name]
                             (count (:inferences (get-in current-matrix [row-name col-name]))))
                           class-names))
                   class-names)}))

(defn get-matrix-data
  [row col]
  (let [{:keys [class-names]} (get-confusion-matrix)
        rich-matrix @confusion-matrix
        row-name (get class-names row)
        col-name (get class-names col)
        entry (get-in rich-matrix [row-name col-name])]
    (when entry
      (let [{:keys [inferences observations]} entry]
        (->> (interleave (map m/emax inferences)
                         observations)
             (partition 2)
             (map vec)
             (sort-by first >))))))


(defn get-confusion-detail
  [{:keys [row col] :as params}]
  (vec (take 50 (map first (get-matrix-data row col)))))



(defn get-confusion-image
  [params]
  (let [{:keys [row col index]} (->> params
                                     (map (fn [[k v]]
                                            [k (clojure.edn/read-string v)]))
                                     (into {}))
        observation (second (nth (get-matrix-data row col) index))
        img (core/mnist-observation->image observation)]
    img))


(def routing-map
  {"confusion-matrix" #'get-confusion-matrix
   "confusion-detail" #'get-confusion-detail
   "confusion-image" #'get-confusion-image})


(defn gate
  []
  (when-not @confusion-matrix
   (reset-confusion-matrix (classification/evaluate-network (core/create-dataset) (core/load-trained-network-desc))))
  (gate/open #'routing-map))
