(ns mnist.core
  (:require [cortex-gpu.nn.train :as gpu-train]
            [cortex-datasets.mnist :as mnist]
            [cortex.optimise :as opt]
            [cortex.nn.description :as desc]
            [cortex.dataset :as ds]
            [cortex-gpu.nn.description :as gpu-desc]
            [clojure.core.matrix :as m]
            [cortex-gpu.nn.cudnn :as cudnn])
  (:gen-class))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defonce normalized-data (future (mnist/normalized-data)))
(def training-data (future (:training-data @normalized-data)))
(def test-data  (future (:test-data @normalized-data)))

(def training-labels (future (mnist/training-labels)))
(def test-labels (future (mnist/test-labels)))

(def all-data (future (vec (concat (m/rows @training-data) (m/rows @test-data)))))
(def all-labels (future (vec (concat @training-labels @test-labels))))

(defrecord MNISTDataset []
  ds/PDataset
  (dataset-name [this] :mnist)
  (shapes [ds]
    [{:name :data
      :shape {:layout ds/planar-image-layout
              :channel-count 1
              :width 28
              :height 28}}
     {:name :labels
      :shape 10}])

  (get-element [this index]
    [(@all-data index) (@all-labels index)])
  (get-elements [this index-seq]
    (mapv #(ds/get-element this %) index-seq))

  (has-indexes? [ds index-type] true)
  (get-indexes [ds index-type]
    (cond
      (= :training index-type) (range 0 (count (m/rows @training-data)))
      :else (range (count (m/rows @training-data)) (+ (count (m/rows @training-data)) (count (m/rows @test-data))))))
  (label-functions [this] [nil nil]))


(defn create-mnist-dataset [] (->MNISTDataset))

(defn learn-mnist
  []
  (with-bindings {#'cudnn/*cudnn-datatype* (float 0.0)}
   (let [n-epochs 50
         batch-size 100
         nndesc [(desc/input 28 28 1)
                 (desc/dropout 0.9)
                 (desc/convolutional 5 0 1 6 :l2-max-constraint 2.0)
                 (desc/max-pooling 2 0 2)
                 (desc/dropout 0.85)
                 (desc/convolutional 5 0 1 6 :l2-max-constraint 2.0)
                 (desc/max-pooling 2 0 2)
                 (desc/dropout 0.85)
                 (desc/convolutional 3 0 1 16 :l2-max-constraint 2.0)
                 (desc/max-pooling 2 2 1 1 1 1)
                 (desc/dropout 0.85)
                 (desc/linear->relu 100 :l2-max-constraint 2.0)
                 (desc/dropout 0.5)
                 (desc/linear->softmax 10)]
         optimizer (opt/adam)
         loss-fn (opt/softmax-loss)
         network-desc (gpu-train/train-next nndesc optimizer loss-fn (create-mnist-dataset) 20 20 [:data] [:labels])])))

(defn -main
  [& args]
  (println (learn-mnist)))
