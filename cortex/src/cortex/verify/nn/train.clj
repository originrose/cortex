(ns cortex.verify.nn.train
  (:require [cortex.nn.layers :as layers]
            [cortex.nn.execute :as execute]
            [cortex.nn.traverse :as traverse]
            [cortex.nn.network :as network]
            [cortex.dataset :as ds]
            [cortex.loss :as loss]
            [cortex-datasets.mnist :as mnist]
            [cortex.optimise :as opt]
            [clojure.test :refer :all]
            [think.resource.core :as resource]))



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
  [(layers/input 28 28 1 :id :input)
   (layers/convolutional 5 0 1 20 :weights {:l1-regularization 0.001})
   (layers/max-pooling 2 0 2)
   (layers/dropout 0.9)
   (layers/relu)
   (layers/local-response-normalization)
   (layers/convolutional 5 0 1 50 :weights {:l2-regularization 0.001})
   (layers/max-pooling 2 0 2)
   (layers/batch-normalization 0.9)
   (layers/linear->relu 500)
   (layers/linear 10)
   (layers/softmax :id :output)])



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


(defn- train-and-get-results
  [context network input-bindings output-bindings
   batch-size dataset optimiser disable-infer? infer-batch-type
   n-epochs map-fn]
  (let [output-id (ffirst output-bindings)]
    (resource/with-resource-context
      (network/print-layer-summary (-> network
                                       network/build-network
                                       traverse/auto-bind-io
                                       traverse/network->training-traversal))
      (as-> (network/build-network network) net-or-seq
        (execute/train context net-or-seq dataset input-bindings output-bindings
                       :batch-size batch-size
                       :optimiser optimiser
                       :disable-infer? disable-infer?
                       :infer-batch-type infer-batch-type)
        (take n-epochs net-or-seq)
        (map map-fn net-or-seq)
        (last net-or-seq)
        (execute/save-to-network context (get net-or-seq :network) {})
        (execute/infer-columns context net-or-seq dataset input-bindings output-bindings
                               :batch-size batch-size)
        (get net-or-seq output-id)))))



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

        loss-fn (loss/mse-loss)
        input-bindings [(traverse/->input-binding :input :data)]
        output-bindings [(traverse/->output-binding :output
                                                    :stream :labels
                                                    :loss loss-fn)]
        results (train-and-get-results context [(layers/input 2 1 1 :id :input)
                                                (layers/linear 1 :id :output)]
                                       input-bindings output-bindings 1 dataset
                                       (opt/adadelta) true nil 5000 identity)
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

        input-bindings [(traverse/->input-binding :input :data)]
        output-bindings [(traverse/->output-binding :output
                                                    :stream :labels
                                                    :loss loss-fn)]
        inference-batch-type :cross-validation
        label-seq (ds/get-batches dataset batch-size inference-batch-type [:labels])
        answers (->> (ds/batches->columnsv label-seq)
                     :labels)
        results (train-and-get-results context mnist-network input-bindings output-bindings batch-size
                                       dataset (opt/adam) false inference-batch-type 4
                                       (fn [{:keys [network inferences] :as entry}]
                                         (println (format "Loss for epoch %s: %s"
                                                          (get network :epoch-count)
                                                          (execute/inferences->node-id-loss-pairs
                                                           network inferences
                                                           (ds/get-batches dataset batch-size
                                                                           inference-batch-type
                                                                           (traverse/get-output-streams
                                                                            network)))))
                                         entry))
        score (loss/evaluate-softmax results answers)]
    (is (> score 0.6))))
