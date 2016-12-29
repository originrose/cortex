(ns cortex.suite.css-styles
  (:require [garden.def :refer [defstylesheet defstyles]]
            [garden.units :refer [px percent]]
            [garden.selectors :refer [nth-child]]))


(def title
  [:div.title {:font-size (px 40)
               :margin [[(px 40) 0 (px 40) (px 80)]]}])

(def confusion-matrix
  [:div.confusion-matrix {:display :inline-block
                          :vertical-align :top}
   [:table.confusion-matrix {:border-spacing 0
                             :border-collapse :collapse}
    [:td.class-header {:text-align :center}
     [:&.top {:padding-bottom (px 40)
              :font-weight :bold}]
     [:&.left {:font-weight :bold
               :transform "rotate(-90deg)"
               :-webkit-transform "rotate(-90deg)"
               :-moz-transform "rotate(-90deg)"
               :-ms-transform "rotate(-90deg)"
               :-o-transform "rotate(-90deg)"}]]
    [:td.item-header {:text-align :center}
     [:&.left-row {:text-align :right
                   :padding-right (px 25)
                   :border-right [[:solid (px 1) :#222]]}]
     [:&.top-row {:padding-bottom (px 12)
                  :border-bottom [[:solid (px 1) :#222]]}]]
    [:td.item-value {:padding (px 12)
                     :text-align :center
                     :cursor :pointer
                     :color :#228}
     [:&:hover {:background-color :#e0e0e0}]]]
   [:td.bottom-row {:padding-left (px 10)}
    [:&.last {:padding-right (px 25)
              :border-right  "solid 1px #222"
              :text-align :right}]]
   [:td.end-column {:padding-left (px 10)}]])

(def confusion-detail
  [:div.confusion-detail {:display :inline-block
                          :margin [[(px 50) 0 0 (px 80)]]}
   [:td.detail {:padding [[0 (px 20) (px 20) 0]]}]])

(def dataset
  [:div.dataset {:margin-left (px 80)}
   [:div.batch-column {:margin-right (px 50)
                       :display :inline-block}
    [:div.batch-name {:margin-bottom (px 10)
                      :border-bottom :solid
                      :font-weight :bold}]]])
(def mnist-styles
  [[:body {:font-family "Monospace"
           :font-size (px 14)
           :margin [[(px 60) (px 30)]]}
    [title]
    [:div.confusion {:margin-bottom (px 60)}
     [confusion-matrix]
     [confusion-detail]]
    [dataset]]])
