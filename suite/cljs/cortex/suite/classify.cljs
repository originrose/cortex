(ns cortex.suite.classify
  (:require-macros [cljs.core.async.macros :refer [go]])
  (:require [cljs.core.async :as async :refer [<!]]
            [reagent.core :refer [atom]]
            [think.gate.core :as gate]
            [think.gate.model :as model]))


(enable-console-print!)



(defn confusion-matrix-component
  [{:keys [class-names matrix]} value-click-handler]
  (let [num-classes (count class-names)]
    [:table.confusion-matrix
     [:tbody
      [:tr [:td.class-header.top {:col-span (str (+ num-classes 3))}
            "predicted"]]
      [:tr [:td.class-header.left {:row-span (str (+ num-classes 3))} "actual"]]
      [:tr (doall (map (fn [cls-name]
                         ^{:key cls-name}
                         [:td.item-header.top-row cls-name])
                       (concat [""] class-names ["correct/actual"])))]
      (doall (map (fn [row-idx]
                    ^{:key row-idx}
                    [:tr [:td.item-header.left-row (get class-names row-idx)]
                     (doall (map (fn [col-idx]
                                   ^{:key col-idx}
                                   [:td.item-value {:on-click (fn [_]
                                                                (value-click-handler row-idx
                                                                                     col-idx))}
                                    (get-in matrix [row-idx col-idx])])
                                 (range num-classes)))
                     (let [row-total (reduce + (map #(get-in matrix [row-idx %]) (range num-classes)))
                           class-correct (get-in matrix [row-idx row-idx])]
                       [:td.end-column (.toFixed (/ class-correct row-total) 3)])])
                  (range num-classes)))
      [:tr [:td.bottom-row.last "correct/" [:br] "predicted"]
       (doall (map (fn [col-idx]
                     (let [col-total (reduce + (map #(get-in matrix [% col-idx]) (range num-classes)))
                           class-correct (get-in matrix [col-idx col-idx])]
                       ^{:key col-idx}
                       [:td.bottom-row (.toFixed (/ class-correct col-total) 3)]))
                   (range num-classes)))
       (let [complete-total (reduce + (flatten matrix))
             diagonal-total (reduce + (map #(get-in matrix [% %]) (range num-classes)))]
         [:td.bottom-row (.toFixed (/ diagonal-total complete-total) 3)])]]]))


(defn confusion-detail-component
  [row col confidence-seq]
  [:table.confusion-detail
   [:tbody
    (doall (map-indexed
            (fn [idx detail-seq]
              ^{:key idx}
              [:tr (doall (map (fn [[idx confidence]]
                                 ^{:key idx}
                                 [:td.detail
                                  [:img {:src (str "confusion-image?row=" row
                                                   "&col=" col "&index=" idx)}]
                                  [:div.detail-confidence
                                   (.toFixed confidence 3)]])
                               detail-seq))])
            (partition 10 10 [] (map-indexed vector confidence-seq))))]])


(defn display-dataset
  [dataset]
  [:div.dataset
   (doall (map (fn [[batch-type {:keys [labels]}]]
                 ^{:key batch-type}
                 [:div.batch-column
                  [:div.batch-name (name batch-type)]
                  [:table
                   [:tbody
                    (doall (map-indexed
                            (fn [idx label-seq]
                              ^{:key idx}
                              [:tr (doall (map (fn [[label-idx label]]
                                                 ^{:key label-idx}
                                                 [:td.dataset-entry
                                                  [:img {:src (str "dataset-image?batch-type="
                                                                   batch-type
                                                                   "&index=" label-idx)}]
                                                  [:div.dataset-label  label]])
                                               label-seq))])
                            (partition 5 5 [] (map-indexed vector labels))))]]])
               dataset))])


(defn confusion-matrix-update
  [confusion-atom dataset-atom]
  (go (let [new-matrix (<! (model/put "confusion-matrix"))
            current-matrix (get @confusion-atom :confusion-matrix)]
        (when-not (= (get new-matrix :update-index)
                     (get current-matrix :update-index))
          (swap! confusion-atom assoc :confusion-matrix new-matrix)
          (swap! confusion-atom dissoc :detail))))
  (go (let [dataset-data (<! (model/put "dataset-data"))
            current-data @dataset-atom]
        (when-not (= (get dataset-data :update-index)
                     (get current-data :update-index))
          (reset! dataset-atom dataset-data)))))


(defn classify-component
  [& args]
  (let [confusion-atom (atom {})
        dataset-atom (atom {})]
    (js/setInterval #(confusion-matrix-update confusion-atom dataset-atom) 1000)
    (fn [& args]
      [:div.classification
       [:div.title "CONFUSION MATRIX"]
       (if-let [confusion-matrix (get @confusion-atom :confusion-matrix)]
         [:div.confusion
          [:div.confusion-matrix
           [confusion-matrix-component confusion-matrix
            (fn [row-idx col-idx]
              (go (swap! confusion-atom assoc :detail
                         {:row row-idx
                          :col col-idx
                          :confidence (<! (model/put "confusion-detail" {:row row-idx
                                                                         :col col-idx}))})))]]
          (when-let [{:keys [row col confidence]} (get @confusion-atom :detail)]
            [:div.confusion-detail
             [confusion-detail-component row col confidence]])]
         [:div "loading matrix"])
       [:div.title "DATASET"]
       (if-let [dataset-data @dataset-atom]
         [display-dataset (dataset-data :dataset)]
         [:div "loading dataset"])])))
