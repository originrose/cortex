(ns cortex.verify.nn.train
  (:require [cortex.nn.layers :as layers]
            [cortex.nn.execute :as execute]
            [cortex.nn.traverse :as traverse]
            [cortex.nn.build :as build]
            [cortex.dataset :as ds]
            [cortex.loss :as loss]
            [cortex-datasets.mnist :as mnist]
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


(def mnist-network
  [(layers/input 28 28 1)
   (layers/convolutional 5 0 1 20)
   (layers/max-pooling 2 0 2)
   (layers/dropout 0.9)
   (layers/relu)
   ;(layers/local-response-normalization)
   (layers/convolutional 5 0 1 50)
   (layers/max-pooling 2 0 2)
   ;(layers/batch-normalization 0.9)
   (layers/linear->relu 500)
   (layers/linear->softmax 10)])



(defonce training-data (future (mnist/training-data)))
(defonce training-labels (future (mnist/training-labels)))
(defonce test-data (future (mnist/test-data)))
(defonce test-labels (future (mnist/test-labels)))


(defn mnist-dataset
  [& {:keys [data-transform-function]
      :or {data-transform-function identity}}]
  (let [data (mapv data-transform-function (concat @training-data @test-data))
        labels (vec (concat @training-labels @test-labels))
        num-training-data (count @training-data)
        total-data (+ num-training-data (count @test-data))
        training-split (double (/ num-training-data
                                  total-data))
        cv-split (- 1.0 training-split)]
    (ds/create-in-memory-dataset {:data {:data data
                                         :shape (ds/create-image-shape 1 28 28)}
                                  :labels {:data labels
                                           :shape 10}}
                                 (ds/create-index-sets total-data
                                                       :training-split training-split
                                                       :cv-split cv-split
                                                       :randimize? false))))




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
                           :optimiser (layers/adadelta)
                           :disable-infer true)
        results (-> (execute/infer context net dataset input-bindings output-bindings :batch-size 1)
                    :test)
        mse (loss/average-loss loss-fn results CORN-LABELS)]
    (is (< mse 25))))


(defn train-mnist
  [context]
  (let [batch-size 10
        n-epochs 4
        epoch-counter (atom 0)
        ;;Don't do this for real.
        max-sample-count 100
        loss-fn (loss/softmax-loss)
        dataset (->> (mnist-dataset)
                     (ds/take-n max-sample-count))
        input-bindings {:input-1 :data}
        output-bindings {:softmax-1 {:stream :labels
                                     :loss loss-fn}}
        inference-batch-type :cross-validation
        label-seq (ds/get-batches dataset batch-size inference-batch-type [:labels])
        answers (->> (ds/batches->columns label-seq)
                     :labels)
        net (execute/train context (build/build-network mnist-network)
                           dataset input-bindings output-bindings
                           (fn [{:keys [network inferences dataset-bindings]}]
                             (println (format "Losses for epoch %s: %s"
                                              (get network :epoch-count)
                                              (vec (execute/inferences->node-id-loss-pairs
                                                    inferences label-seq dataset-bindings))))
                             (< (swap! epoch-counter inc) n-epochs))
                           :batch-size batch-size
                           :infer-batch-type inference-batch-type)
        results (->> (execute/infer context net dataset input-bindings output-bindings
                                    :batch-size batch-size
                                    :infer-batch-type inference-batch-type)
                     :softmax-1)

        score (loss/evaluate-softmax results answers)]
    (println score)
    (is (> score 0.6))))
