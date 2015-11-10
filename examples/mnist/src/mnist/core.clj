(ns mnist.core
  (:require
    [clojure.core.matrix :as mat]
    [thinktopic.datasets.mnist :as mnist]
    [thinktopic.cortex.util :as util]
    [thinktopic.cortex.network :as net]
    [thinktopic.fmri.core :as fmri]))

(def trained* (atom nil))

(defn mnist-labels
  [class-labels]
  (let [n-labels (count class-labels)
        labels (mat/zero-array [n-labels 10])]
    (doseq [i (range n-labels)]
      (mat/mset! labels i (nth class-labels i) 1.0))
    labels))

(def MNIST-LABELS (mnist-labels @mnist/label-store))

; FMRI Ideas
; - publish a dataset for browsing
; - publish a network
;  * view features
;  * view loss
;  * view train and test performance

(defn setup
  []
  (print "setup...")
  (let [start-time (util/timestamp)
        training-data @mnist/data-store
        [n-inputs input-width] (mat/shape training-data)
        training-data (mat/submatrix training-data 0 100 0 input-width)
        training-data (map #(mat/broadcast % [1 input-width]) training-data)
        training-labels (mnist-labels @mnist/label-store)
        test-data @mnist/test-data-store
        test-labels (mnist-labels @mnist/test-label-store)
        net (or net (net/sequential-network
                      [(net/linear-layer :n-inputs 784 :n-outputs 30)
                       (net/sigmoid-activation 30)
                       (net/linear-layer :n-inputs 30 :n-outputs 10)
                       (net/sigmoid-activation 10)]))
        machine (fmri/machine)]
    (println " [" (util/ms-elapsed start-time (util/timestamp)) setup-time "ms]")
    (fmri/register-dataset! machine "mnist" training-data training-labels)
    (fmri/register-network! machine "mnist" net)
    {:net net
     :fmri machine
     :training {:data training-data :labels training-labels}
     :testing {:data test-data :labels test-labels}}))

(defn run-experiment
  [{:keys [net training testing]}]
  (let [n-epochs 1
        learning-rate 3.0
        momentum 0.9
        batch-size 10
        loss-fn (net/quadratic-loss)
        optimizer (net/sgd-optimizer net loss-fn learning-rate momentum)]
    (fmri/register-experiment! "mnist" optimizer)
    (println "training network...")
    (net/train-network optimizer n-epochs batch-size (:data training) (:labels trainging))

    (println "evaluating network...")
    (let [[results score] (net/evaluate net (:data testing)  (:labels trainging))
          label-count (first (mat/shape test-labels))
          score-percent (float (/ score label-count))]
      (reset! trained* net)
      (println (format "MNIST Score: %f [%d of %d]" score-percent score label-count)))))

(defn -main []
  (let [experiment (setup)]
    (run-experiment experiment)))

