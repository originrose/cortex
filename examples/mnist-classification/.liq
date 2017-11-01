
(ns user
  (:require [dk.salza.liq.editor :as editor]
            [dk.salza.liq.clojureutil :as clojureutil]
            [dk.salza.liq.tools.cshell :refer :all]
            [dk.salza.liq.apps.textapp :as textapp]
            [dk.salza.liq.logging :as logging]
            [clojure.java.io :as io]
            [cortex.datasets.mnist :as mnist]
            [mikera.image.core :as i]
            [think.image.image :as image]
            [think.image.patch :as patch]
            [think.image.data-augmentation :as image-aug]
            [cortex.nn.layers :as layers]
            [clojure.core.matrix.macros :refer [c-for]]
            [clojure.core.matrix :as m]
            [cortex.experiment.classification :as classification]
            [cortex.experiment.train :as train]
            [cortex.nn.network :as network]
            [cortex.nn.execute :as execute]
            [cortex.util :as util]
            [cortex.experiment.util :as experiment-util])
  (:import [java.io File]))

(editor/find-file (.getCanonicalPath (clojure.java.io/file "./walkthrough.md")))
