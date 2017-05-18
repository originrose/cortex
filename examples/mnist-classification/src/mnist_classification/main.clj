(ns mnist-classification.main
  (:require [clojure.tools.cli :refer [parse-opts]])
  (:gen-class))

(def cli-options
  [["-f" "--force-gpu value" "Force GPU to be used"
    :id :force-gpu?
    :parse-fn #(Boolean/parseBoolean %)
    :default true]
   [ "-l" nil
    :long-opt "--live-updates"
    :id :live-updates?
    :default false]])

(defn -main
  [& args]
  (println "Welcome! Please wait while we compile some Clojure code...")
  (let [argmap (parse-opts args cli-options)]
    (require 'mnist-classification.core)
    ((resolve 'mnist-classification.core/train-forever-uberjar)
     (:options argmap))))
