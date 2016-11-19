(ns think.compute.verify.nn.train
  (:require [clojure.test :refer :all]
            [think.compute.math :as math]
            [think.compute.verify.utils :as utils]
            [think.compute.nn.backend :as nn-backend]
            [think.compute.nn.train :as train]
            [think.compute.nn.layers :as layers]
            [think.compute.optimise :as opt]
            [think.compute.batching-system :as batch]
            [cortex.dataset :as ds]
            [cortex.nn.protocols :as cp]
            [clojure.core.matrix :as m]
            [think.compute.verify.nn.mnist :as mnist]
            [think.compute.nn.evaluate :as nn-eval]
            [cortex.nn.description :as desc]
            [think.compute.nn.description :as compute-desc]))


(defn mse-loss
  [backend num-batch-items layer]
  (-> (opt/mse-loss)
      (opt/setup-loss backend num-batch-items (cp/output-size layer))))


(defn test-train-step
  [backend]
  (let [num-batch-items 10
        weights (nn-backend/array backend [[1 2] [3 4]])
        bias (nn-backend/array backend [0 10])
        input (nn-backend/array backend (flatten (repeat num-batch-items [1 2])) num-batch-items)
        layer (layers/->Linear backend weights bias nil)
        layer (cp/setup layer num-batch-items)
        loss-fn (-> (mse-loss backend num-batch-items layer))
        train-config {:network layer :loss-fn [loss-fn] :backend backend}
        layer (:network
               (train/train-step train-config [input] [(nn-backend/array backend
                                                                      (flatten
                                                                       (repeat num-batch-items [4 19])))]))
        output (cp/output layer)
        output-data (nn-backend/to-double-array backend output)
        weight-gradient (vec (nn-backend/to-double-array backend (:weight-gradient layer)))
        bias-gradient (vec (nn-backend/to-double-array backend (:bias-gradient layer)))
        input-gradient (vec (nn-backend/to-double-array backend (:input-gradient layer)))]
    (is (= (map double (flatten (repeat num-batch-items [5 21])))
           (m/eseq output-data)))
    (is (m/equals (mapv #(* % num-batch-items) [1 2 2 4]) weight-gradient))
    (is (m/equals (mapv #(* % num-batch-items) [1 2]) bias-gradient))
    (is (m/equals (flatten (repeat num-batch-items [7 10])) input-gradient))))


(defn test-optimise
  [backend]
  (let [num-batch-items 10
        weights (nn-backend/array backend [[1 2] [3 4]])
        bias (nn-backend/array backend [0 10])
        input (nn-backend/array backend (flatten (repeat num-batch-items [1 2])) num-batch-items)
        layer (layers/->Linear backend weights bias nil)
        layer (cp/setup layer num-batch-items)
        train-config (-> {:network layer
                          :loss-fn [(mse-loss backend num-batch-items layer)]
                          :optimiser (opt/setup-optimiser (opt/adadelta) backend (layers/parameter-count layer))
                          :batch-size num-batch-items
                          :backend backend}
                         (train/train-step [input] [(nn-backend/array backend
                                                      (flatten (repeat num-batch-items [4 19])))])
                         train/optimise)
        layer (:network train-config)
        optimiser (:optimiser train-config)]
    (is (utils/about-there? (seq (nn-backend/to-double-array backend weights))
                                 [0.9955279087656892 1.9955278752252983
                                 2.9955278752252985 3.995527866840083]))
    (is (utils/about-there? (seq (nn-backend/to-double-array backend bias))
                                 [-0.004472091234310839 9.995527875225298]))
    (is (utils/about-there? (seq (nn-backend/to-double-array backend (:weight-gradient layer)))
                                 [0 0 0 0]))
    (is (utils/about-there? (seq (nn-backend/to-double-array backend (:bias-gradient layer)))
                                 [0 0]))))


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
  [backend]
  ;;Don't print out per-epoch results.
  (let [net (layers/linear backend 2 1)
        n-epochs 5000
        loss (opt/mse-loss)
        optimizer (opt/adadelta)
        batch-size 1
        corn-indexes (range (count CORN-DATA))
        dataset (ds/create-in-memory-dataset {:data {:data CORN-DATA
                                                     :shape 2}
                                              :labels {:data CORN-LABELS
                                                       :shape 1}}
                                             (ds/create-index-sets (count CORN-DATA)
                                                                   :training-split 1.0
                                                                   :randomize? false))
        net (cp/setup net batch-size)
        net (train/train net optimizer dataset [:data] [[:labels loss]] n-epochs
                         :epoch-train-filter nil)
        ;;First here because we want the results that correspond to the network's *first* output
        results (first (train/run net dataset [:data]))
        mse (opt/evaluate-mse results CORN-LABELS)]
    (is (< mse 25))))


(defn layer->description
  [backend]
  (let [[network dataset] (mnist/train-mnist-network backend {:max-sample-count 100})
        score (nn-eval/evaluate-softmax network dataset [:data])
        network-desc (desc/network->description network)
        _ (comment
            (clojure.pprint/pprint
             (mapv #(dissoc % :weights :bias :scale :variances :means)
                   (flatten network-desc))))
        new-network (compute-desc/build-and-create-network network-desc backend 10)
        new-score (nn-eval/evaluate-softmax new-network dataset [:data])]
    (is (utils/about-there? score new-score))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; y = mx + b test
(defn net->m-and-b
  [backend net]
  (->> net
       (layers/parameters)
       (map (partial nn-backend/to-double-array backend))
       (map vec)
       (flatten)))

(defn learn-m-and-b
  [backend description m b]
  (let [net (compute-desc/build-and-create-network description backend 1)
        n 200
        inputs (for [_ (range n)] (- (rand 200) 100))
        outputs (for [x inputs] (+ b (* m x)))
        data (mapv vector inputs)
        labels (mapv vector outputs)
        dataset (ds/create-in-memory-dataset
                 {:data {:data data :shape 1}
                  :labels {:data labels :shape 1}}
                 (ds/create-index-sets n :training-split 1.0))]
    (train/train net (opt/adam) dataset [:data] [[:labels (opt/mse-loss)]] 200
                 :epoch-train-filter nil)
    net))

(defn test-simple-learning-attenuation
  [backend]
  (let [m 2
        b 3
        description [(desc/input 1)
                     (desc/linear 1)]
        net (learn-m-and-b backend description m b)]
    (let [[learned-m learned-b] (net->m-and-b backend net)]
      (is (> 0.1 (Math/abs (- m learned-m))))
      (is (> 0.1 (Math/abs (- b learned-b)))))
    (let [m -3
          b -2
          frozen-description (->> net
                                  (desc/network->description)
                                  (mapv #(assoc % :learning-attenuation 0.0)))
          net (learn-m-and-b backend frozen-description m b)]
      (let [[learned-m learned-b] (net->m-and-b backend net)]
        (is (< 0.1 (Math/abs (- m learned-m))))
        (is (< 0.1 (Math/abs (- b learned-b))))))))
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; softmax test
(defn- mode
  [l]
  (ffirst (sort-by (comp count second) > (group-by identity l))))

(defn- count->bit-vector
  "Create a bit-vector of length = max-count with the 1 at position c"
  [max-count c]
  (let [c (min (dec max-count) c)]
    (-> (take max-count (repeatedly (constantly 0)))
        (vec)
        (assoc c 1))))

(defn test-softmax-channels
  [backend]
  (let [input-dim 4
        output-dim 2
        classes 3
        description [(desc/input input-dim)
                     (desc/linear->softmax
                      (* output-dim classes) :output-channels classes)]
        batch-size 5
        net (compute-desc/build-and-create-network description backend batch-size)
        n 1000
        data (vec (repeatedly n (fn [] (repeatedly 4 #(rand (double classes))))))
        labels (mapv #(mapcat (comp (partial count->bit-vector classes) int mode)
                              (partition 2 %))
                     data)
        train-dataset (ds/create-in-memory-dataset
                       {:data {:data data :shape input-dim}
                        :labels {:data labels :shape (* output-dim classes)}}
                       (ds/create-index-sets n :training-split 1.0))
        epoch-count 100
        _ (train/train net (opt/adam) train-dataset [:data] [[:labels (opt/softmax-loss :output-channels classes)]] epoch-count
                       :epoch-train-filter nil)
        holdout-dataset (ds/create-in-memory-dataset
                         {:data {:data [[0.1 0.1 2.9 2.9]] :shape input-dim}}
                         (ds/create-index-sets 1 :training-split 1.0))
        holdout-net (compute-desc/build-and-create-network (desc/network->description net) backend 1)
        result (ffirst (train/run holdout-net holdout-dataset [:data]))
        [a b c d e f] result]
    (is (> a b))
    (is (> a c))
    (is (> f d))
    (is (> f e))))
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
