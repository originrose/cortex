(ns thinktopic.cortex.lab.charts
  (:import [org.jfree.chart ChartPanel])
  (:import [java.awt Component Dimension])
  (:import [javax.swing JFrame])
  (:require [mc ui])
  (:use [task core])
  (:use [incanter core stats charts]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* true)

;; =============================================
;; chart display functions

(def window-count (atom 1))

(defn show-chart
  ([incanter-chart]
    (show-chart incanter-chart (str "Chart" (swap! window-count inc))))
  ([incanter-chart title]
    (mc.ui/show-component (ChartPanel. incanter-chart) title)))

;; =============================================
;; example code

(comment
  ; test chart
  (view (function-plot sin -10 10))
  )
