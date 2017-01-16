(ns suite-classification.main
  (:gen-class))

(defn -main
  [& args]
  (require 'suite-classification.core)
  (if (= "live-updates" (first args))
    ((resolve 'suite-classification.core/train-forever))
    ((resolve 'suite-classification.core/train-forever-uberjar))))
