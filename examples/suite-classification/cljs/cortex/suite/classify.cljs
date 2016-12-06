(ns cortex.suite.classify
  (:require-macros [cljs.core.async.macros :refer [go]])
  (:require [cljs.core.async :as async :refer [<!]]
            [reagent.core :refer [atom]]
            [think.gate.core :as gate]
            [think.gate.model :as model]))

(enable-console-print!)

(def state* (atom nil))

(defn interactive-component
  []
  (fn []
    (if-let [confusion-matrix (get @state* :confusion-matrix)]
      (let [{:keys [class-names matrix]} confusion-matrix
            num-classes (count class-names)]
        [:div.confusion
         [:table.confusion-matrix
          [:tbody
           [:tr [:td.class-header {:col-span (str (+ num-classes 3))
                                   :style {:text-align :center}}
                 "predicted"]]
           [:tr [:td.class-header {:row-span (str (+ num-classes 3))} "actual"]]
           [:tr (doall (map (fn [cls-name]
                              ^{:key cls-name}
                              [:td.item-header {:style {:text-align :center}} cls-name])
                            (concat [""] class-names ["correct/actual"])))]
           (doall (map (fn [row-idx]
                         ^{:key row-idx}
                         [:tr [:td.item-header {:style {:text-align :center}} (get class-names row-idx)]
                          (doall (map (fn [col-idx]
                                        ^{:key col-idx}
                                        [:td.item-value {:style {:text-align :center}
                                                         :on-click (fn [_]
                                                                     (go (swap! state* assoc :detail
                                                                                {:row row-idx
                                                                                 :col col-idx
                                                                                 :confidence (<! (model/put "confusion-detail" {:row row-idx
                                                                                                                                :col col-idx}))})))}
                                         (get-in matrix [row-idx col-idx])])
                                      (range num-classes)))
                          (let [row-total (reduce + (map #(get-in matrix [row-idx %]) (range num-classes)))
                                class-correct (get-in matrix [row-idx row-idx])]
                            [:td.end-row (.toFixed (/ class-correct row-total) 3)])])
                       (range num-classes)))
           [:tr [:td.item-header "correct/predicted"]
            (doall (map (fn [col-idx]
                          (let [col-total (reduce + (map #(get-in matrix [% col-idx]) (range num-classes)))
                                class-correct (get-in matrix [col-idx col-idx])]
                            ^{:key col-idx}
                            [:td.end-row (.toFixed (/ class-correct col-total) 3)]))
                        (range num-classes)))
            (let [complete-total (reduce + (flatten matrix))
                  diagonal-total (reduce + (map #(get-in matrix [% %]) (range num-classes)))]
              [:td.total-total (.toFixed (/ diagonal-total complete-total) 3)])]]]
         (when-let [{:keys [row col confidence]} (get @state* :detail)]
           [:table.confusion-detail
            [:tbody
             (doall (map-indexed
                     (fn [idx detail-seq]
                       ^{:key idx}
                       [:tr (doall (map (fn [[idx confidence]]
                                          ^{:key idx}
                                          [:td.detail
                                           [:img {:src (str "confusion-image?row=" row "&col=" col "&index=" idx)}]
                                           [:div.detail-confidence {:style {:text-align :center}}
                                            (.toFixed confidence 3)]])
                                        detail-seq))])
                     (partition 10 10 [] (map-indexed vector confidence))))]])])
      (do (go (swap! state* assoc :confusion-matrix (<! (model/put "confusion-matrix"))))
          [:div "loading matrix"]))))

(think.gate.core/set-component [interactive-component])
