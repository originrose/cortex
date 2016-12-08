(ns cortex.suite.css-styles)

(def styles
  [[:td.class-header {:text-align :center}]
   [:td.item-header {:text-align :center}]
   [:td.item-value {:text-align :center
                    :cursor "pointer"
                    :color "#228"}]
   [:div.batch-column {:display "inline-block"
                       :border "solid #228 2px"
                       :box-sizing "border-box"
                       :vertical-align "top"}]
   [:div.detail-confidence {:text-align "center"}]
   [:div.batch-name {:text-align "center"}]
   [:div.dataset-label {:text-align "center"}]])
