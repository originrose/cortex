(ns cortex.verify.nn.train
  (:require [cortex.nn.layers :as layers]
            [cortex.nn.execute :as execute]
            [cortex.nn.traverse :as traverse]
            [cortex.nn.build :as build]
            [cortex.dataset :as ds]
            [cortex.loss :as loss]
            [clojure.test :refer :all]))



;; Data from: Dominick Salvator and Derrick Reagle
;; Shaum's Outline of Theory and Problems of Statistics and Economics
;; 2nd edition,  McGraw-Hill, 2002, pg 157

;; Predict corn yield from fertilizer and insecticide inputs
;; [corn, fertilizer, insecticide]
(def CORN-DATA
  [[6  4]
   [10  4]
   [12  5]
   [14  7]
   [16  9]
   [18 12]
   [22 14]
   [24 20]
   [26 21]
   [32 24]])


(def CORN-LABELS
  [[40] [44] [46] [48] [52] [58] [60] [68] [74] [80]])

(defn test-corn
  [context]
  (let [epoch-counter (atom 0)
        dataset (ds/create-in-memory-dataset {:data {:data CORN-DATA
                                                     :shape 2}
                                              :labels {:data CORN-LABELS
                                                       :shape 1}}
                                             (ds/create-index-sets (count CORN-DATA)
                                                                   :training-split 1.0
                                                                   :randomize? false))
        n-epochs 5000
        input-bindings {:input :data}
        loss-fn (loss/mse-loss)
        output-bindings {:test {:stream :labels
                                :loss loss-fn}}
        net (build/build-network [(layers/input 2 1 1 :id :input)
                                  (layers/linear 1 :id :test)])
        net (execute/train context net dataset input-bindings output-bindings
                           (fn [& args]
                             (< (swap! epoch-counter inc) n-epochs))
                           :batch-size 1
                           :optimiser (layers/adadelta))
        results (-> (execute/infer context net dataset input-bindings output-bindings :batch-size 1)
                 (get :test))
        mse (loss/average-loss loss-fn results CORN-LABELS)]
    (is (< mse 25))))
