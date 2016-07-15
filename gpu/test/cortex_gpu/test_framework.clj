(ns cortex-gpu.test-framework
  (:require [cortex-gpu.cuda :as cuda]
            [clojure.core.matrix :as m]
            [cortex-gpu.nn.train :as train]
            [cortex-gpu.nn.cudnn :as cudnn]
            [cortex-gpu.nn.layers :as layers]
            [cortex.nn.protocols :as cp]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.protocols :as mp]
            [cortex-datasets.mnist :as mnist]
            [cortex-gpu.nn.batch :as batch]
            [cortex.optimise :as opt]
            [cortex.nn.core :as core]
            [cortex.nn.network :as net]
            [cortex.nn.backends :as b]
            [cortex.nn.description :as desc]
            [resource.core :as resource]
            [cortex-gpu.nn.description :as gpu-desc]
            [clojure.test :refer :all])
  (:import [java.math BigDecimal MathContext]))


(defn with-contexts
  [test-fn]
  (resource/with-resource-context
    (cuda/create-context)
    (cudnn/create-context)
    (test-fn)))

(defn with-resource-context
  [test-fn]
  (resource/with-resource-context
    (test-fn)))


(defn abs-diff
  [a b]
  (m/esum (m/abs (m/sub a b))))


(def epsilon 1e-6)

(defn about-there?
  ([a b eps]
   (< (abs-diff a b) eps))
  ([a b]
   (about-there? a b epsilon)))



(defn layer-items
  [layer-list weight-key bias-key]
  (let [layer-list (filter #(and (get % weight-key)
                                 (get % bias-key))
                           layer-list)]
    (vec (mapcat #(vector (get % weight-key)
                          (get % bias-key))
                 layer-list))))


(defn compare-cpu-gpu-layers
  "Compare cpu and gpu networks.  Indicate briefly any differences
and return buffers so you can inspect the results.  Note that there
are certainly differences in numerical operations between the gpu and cpu
so even correctly written layers will differ somewhat w/r/t cpu vs. gpu."
  [cpu-network gpu-network training-data training-labels]
  (resource/with-resource-context
   (let [total-count (count training-data)
         batch-count total-count
         loss-fn (opt/mse-loss)
         cpu-network (reduce (fn [network [input answer]]
                               (net/train-step input answer network loss-fn))
                             cpu-network
                             (map vector training-data training-labels))
         gpu-network (batch/setup gpu-network total-count batch-count
                                  (cp/input-size gpu-network)
                                  (cp/output-size gpu-network))
         gpu-dataset (batch/load-dataset-to-gpu [training-data training-labels])
         network (batch/upload-randomized-indexes gpu-network total-count)
         input-buffer (get gpu-network batch/data-keyword)
         output-buffer (get gpu-network batch/label-keyword)
         gpu-network (batch/load-batch-buffer gpu-network (gpu-dataset 0) 0 input-buffer)
         gpu-network (batch/load-batch-buffer gpu-network (gpu-dataset 1) 0 output-buffer)
         gpu-network (train/train-step gpu-network input-buffer output-buffer)
         gpu-outputs (mapv (comp cudnn/to-double-array cp/output) (:layers gpu-network))
         cpu-outputs (mapv (comp mp/to-double-array cp/output) (:modules cpu-network))
         gpu-gradients (mapv cudnn/to-double-array (layers/gradients gpu-network))
         cpu-parameters (mapv mp/to-double-array (layer-items
                                                  (:modules cpu-network)
                                                  :weights :bias))
         cpu-gradients (mapv mp/to-double-array (layer-items
                                                 (:modules cpu-network)
                                                 :weight-gradient :bias-gradient))
         item-count (count gpu-outputs)]
     (doseq [item-idx (range item-count)]
       (when-not (about-there? (gpu-outputs item-idx)
                               (cpu-outputs item-idx))
         (println (format "outputs at idx %d differ" item-idx)))
       (when-not (about-there? (gpu-gradients item-idx)
                               (cpu-gradients item-idx))
         (println (format "Gradients at idx %d differ" item-idx))))
     [{:gradients {:gpu gpu-gradients :cpu cpu-gradients}
       :output {:gpu gpu-outputs :cpu cpu-outputs}}])))


(defn equal-count
  ^long [lhs rhs]
  (count (filter #(= (% 0) (% 1))
                 (map vector lhs rhs))))


(defmacro def-double-float-test
  "Define a test that tests both the double and floating point versions of a function"
  [test-name & body]
  (let [double-name (str test-name "-d")
        float-name (str test-name "-f")]
    `(do
       (deftest ~(symbol double-name)
         ~@body)
       (deftest ~(symbol float-name)
         (with-bindings {#'cudnn/*cudnn-datatype* (float 0.0)}
           ~@body)))))


(defn round-to-sig-figs
  ^double [^double lhs ^long num-sig-figs]
  (-> (BigDecimal. lhs)
      (.round (MathContext. num-sig-figs))
      (.doubleValue)))


(defn sig-fig-equal?
  ([^double lhs ^double rhs ^long num-sig-figs]
   (= (round-to-sig-figs lhs num-sig-figs)
      (round-to-sig-figs rhs num-sig-figs)))
  ([^double lhs ^double rhs]
   (sig-fig-equal? lhs rhs 4)))
