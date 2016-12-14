(ns css.styles
  (:require [cortex.suite.css-styles :as suite-styles]
            [garden.def :refer [defstylesheet defstyles]]
            [garden.units :refer [px percent]]
            [garden.selectors :refer [nth-child]]))

(defstyles styles suite-styles/mnist-styles)
