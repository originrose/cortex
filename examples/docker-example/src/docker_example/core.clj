(ns docker-example.core
  (:require [cortex.nn.layers :as layers]
            [cortex.nn.execute :as execute]
            [cortex.nn.network :as network]
            [cortex.verify.nn.data :refer [CORN-DATA CORN-LABELS]])
  (:gen-class))

(defn train-network
  []
  (let [corn-dataset (->> (mapv (fn [d l] {:data d :labels l})
                                CORN-DATA CORN-LABELS)
                          (repeat 1000)
                          (apply concat))]
    (loop [network (network/linear-network
                     [(layers/input 2 1 1 :id :data)
                      (layers/linear 1 :id :labels)])
           epoch 0]
      (if (> 2 epoch)
        (recur (:network (execute/train network corn-dataset :batch-size 10)) (inc epoch))
        network))))

(defn -main
  "Run a simple regression network."
  [& args]
  (let [_ (println "Training network...")
        network (train-network)
        _ (println "Done.")]
    (System/exit 0)))
