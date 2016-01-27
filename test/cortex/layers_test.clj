(ns cortex.layers_test
  "Tests for behaviour of standard layer implementations"
  (:use [clojure.test])
  (:use [cortex.core])
  (:require [clojure.core.matrix :as m]
            [cortex.layers :as layers]
            [cortex.util :as util]
            [clojure.test.check :as sc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [clojure.test.check.clojure-test :refer (defspec)]))
