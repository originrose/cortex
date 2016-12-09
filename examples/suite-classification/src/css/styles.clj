(ns css.styles
  (:require [cortex.suite.css-styles :as suite-styles]
            [garden.def :refer [defstylesheet defstyles]]))

(defstyles styles suite-styles/styles)
