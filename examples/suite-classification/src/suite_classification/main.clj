(ns suite-classification.main
  (:gen-class))

(defn -main
  [& args]
  (require 'suite-classification.core)
  ((resolve 'suite-classification.core/train-forever-uberjar)))
