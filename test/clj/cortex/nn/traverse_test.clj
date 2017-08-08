(ns cortex.nn.traverse-test
  (:require [clojure.test :refer :all]
            [clojure.data :as data]
            [clojure.core.matrix :as m]
            [cortex.graph :as graph]
            [cortex.loss.core :as loss]
            [cortex.nn.traverse :as traverse]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
            [cortex.verify.nn.data :refer [CORN-DATA CORN-LABELS]]))


(def mnist-basic
  [(layers/input 28 28 1)
   (layers/linear 200)
   (layers/relu)
   (layers/linear 10)
   (layers/softmax)])


(def mnist-description-with-toys
  [(layers/input 28 28 1 :id :data)
   (layers/multiplicative-dropout 0.1)
   (layers/convolutional 5 0 1 20 :weights {:l1-regularization 0.001})
   (layers/max-pooling 2 0 2)
   (layers/relu)
   (layers/dropout 0.75)
   (layers/convolutional 5 0 1 50 :l2-regularization 0.01)
   (layers/max-pooling 2 0 2)
   (layers/relu)
   (layers/dropout 0.75)
   (layers/batch-normalization)
   (layers/linear 500) ;;If you use this description put that at 1000
   (layers/relu :id :feature :center-loss {:labels {:stream :output}
                                           :label-indexes {:stream :output}
                                           :label-inverse-counts {:stream :output}
                                           :lambda 0.05
                                           :alpha 0.9})
   (layers/dropout 0.5)
   (layers/linear 10)
   (layers/softmax :id :output)])


(defn minimal-diff
  [lhs rhs]
  (->> (data/diff lhs rhs)
       (take 2)
       vec))


(defn build-big-description
  []
  (network/linear-network mnist-description-with-toys))


(deftest big-description
  (let [network (build-big-description)
        training-traversal (traverse/training-traversal network)
        inference-traversal (traverse/inference-traversal network)]
    ;;Adding in the parameters required for the center loss centers.  10 * 500 = 5000
    ;;extra parameters to a network with 434280 parameters
    (is (= 439280 (graph/parameter-count (get network :compute-graph))))
    (is (= 439280 (->> (get-in network [:compute-graph :buffers])
                       (map (comp m/ecount :buffer second))
                       (reduce +))))

    (is (= :node-argument (-> (network/network->graph network)
                              (graph/get-node :l1-regularization-1)
                              (graph/get-node-argument :output)
                              (get :type))))

    (is (= [nil nil]
           (minimal-diff
            [{:id :dropout-1, :incoming [{:stream :data}], :outgoing [{:id :dropout-1}]}
             {:id :convolutional-1,
              :incoming [{:id :dropout-1}],
              :outgoing [{:id :convolutional-1}]}
             {:id :max-pooling-1, :incoming [{:id :convolutional-1}], :outgoing [{:id :max-pooling-1}]}
             {:id :relu-1, :incoming [{:id :max-pooling-1}], :outgoing [{:id :relu-1}]}
             {:id :dropout-2, :incoming [{:id :relu-1}], :outgoing [{:id :dropout-2}]}
             {:id :convolutional-2, :incoming [{:id :dropout-2}], :outgoing [{:id :convolutional-2}]}
             {:id :max-pooling-2, :incoming [{:id :convolutional-2}], :outgoing [{:id :max-pooling-2}]}
             {:id :relu-2, :incoming [{:id :max-pooling-2}], :outgoing [{:id :relu-2}]}
             {:id :dropout-3, :incoming [{:id :relu-2}], :outgoing [{:id :dropout-3}]}
             {:id :batch-normalization-1, :incoming [{:id :dropout-3}], :outgoing [{:id :batch-normalization-1}]}
             {:id :linear-1, :incoming [{:id :batch-normalization-1}], :outgoing [{:id :linear-1}]}
             {:id :feature, :incoming [{:id :linear-1}], :outgoing [{:id :feature}]}
             {:id :dropout-4, :incoming [{:id :feature}], :outgoing [{:id :dropout-4}]}
             {:id :linear-2, :incoming [{:id :dropout-4}], :outgoing [{:id :linear-2}]}
             {:id :output, :incoming [{:id :linear-2}], :outgoing [{:id :output}]}]
           (get training-traversal :forward))))
    (is (= [nil nil]
           (minimal-diff
            {{:id :batch-normalization-1} {:dimension {:channels 50, :height 4, :width 4}},
             {:id :convolutional-1} {:dimension {:channels 20, :height 24, :width 24}},
             {:id :convolutional-2} {:dimension {:channels 50, :height 8, :width 8}},
             {:id :dropout-1} {:dimension {:channels 1, :height 28, :width 28}},
             {:id :dropout-2} {:dimension {:channels 20, :height 12, :width 12}},
             {:id :dropout-3} {:dimension {:channels 50, :height 4, :width 4}},
             {:id :dropout-4} {:dimension {:channels 1, :height 1, :width 500}},
             {:id :feature} {:dimension {:channels 1, :height 1, :width 500}},
             {:id :linear-1} {:dimension {:channels 1, :height 1, :width 500}},
             {:id :linear-2} {:dimension {:channels 1, :height 1, :width 10}},
             {:id :max-pooling-1} {:dimension {:channels 20, :height 12, :width 12}},
             {:id :max-pooling-2} {:dimension {:channels 50, :height 4, :width 4}},
             {:id :output} {:dimension {:channels 1, :height 1, :width 10}},
             {:id :relu-1} {:dimension {:channels 20, :height 12, :width 12}},
             {:id :relu-2} {:dimension {:channels 50, :height 4, :width 4}},
             {:stream :data} {:dimension {:channels 1, :height 28, :width 28}}
             {:stream :output} {:dimension {:channels 1, :height 1, :width 10}}
             {:stream {:stream :output,
                       :augmentation :cortex.loss.center/labels->inverse-counts}} {:dimension {}},
             {:stream
              {:stream :output,
               :augmentation :cortex.loss.center/labels->indexes}} {:dimension {}}}
            (get training-traversal :buffers))))
    ;;Using set below to make the output order independent.  Loss terms are added so the definition
    ;;of the loss function is independent of order.
    (is (= [nil nil]
           (minimal-diff
            (set [{:type :softmax-loss,
                    :output {:type :node-output, :node-id :output},
                    :labels {:type :stream, :stream :output},
                   :id :softmax-loss-1}
                  {:type :center-loss,
                   :alpha 0.9,
                   :labels {:stream :output},
                   :label-indexes {:stream :output},
                   :label-inverse-counts {:stream :output},
                   :lambda 0.05,
                   :output {:type :node-output, :node-id :feature},
                   :id :center-loss-1,
                   :centers {:buffer-id :center-loss-1-centers-1}}
                  {:type :l2-regularization,
                   :lambda 0.01,
                   :output {:type :node-output, :node-id :convolutional-2},
                   :id :l2-regularization-1}
                  {:type :l1-regularization,
                   :lambda 0.001,
                   :output
                   {:type :node-argument, :node-id :convolutional-1, :argument :weights},
                   :id :l1-regularization-1}])
            (network/loss-function network))))

    (is #{{:type :softmax-loss,
           :output {:type :node-output, :node-id :output},
           :labels {:type :stream, :stream :output},
           :id :softmax-loss-1}
          {:type :center-loss,
           :alpha 0.9,
           :labels {:stream :output},
           :label-indexes {:stream :output},
           :label-inverse-counts {:stream :output},
           :lambda 0.05,
           :output {:type :node-output, :node-id :feature},
           :id :center-loss-1,
           :centers {:buffer-id :center-loss-1-centers-1}}
          {:type :l2-regularization,
           :lambda 0.01,
           :output {:type :node-output, :node-id :convolutional-2},
           :id :l2-regularization-1}
          {:type :l1-regularization,
           :lambda 0.001,
           :output
           {:type :node-argument, :node-id :convolutional-1, :argument :weights},
           :id :l1-regularization-1}}
            (traverse/gradient-loss-function network training-traversal))

    (is (= [nil nil]
           (minimal-diff
            [{:id :convolutional-1, :incoming [{:stream :data}], :outgoing [{:id :convolutional-1}]}
             {:id :max-pooling-1, :incoming [{:id :convolutional-1}], :outgoing [{:id :max-pooling-1}]}
             {:id :relu-1, :incoming [{:id :max-pooling-1}], :outgoing [{:id :relu-1}]}
             {:id :convolutional-2, :incoming [{:id :relu-1}], :outgoing [{:id :convolutional-2}]}
             {:id :max-pooling-2, :incoming [{:id :convolutional-2}], :outgoing [{:id :max-pooling-2}]}
             {:id :relu-2, :incoming [{:id :max-pooling-2}], :outgoing [{:id :relu-2}]}
             {:id :batch-normalization-1, :incoming [{:id :relu-2}], :outgoing [{:id :batch-normalization-1}]}
             {:id :linear-1, :incoming [{:id :batch-normalization-1}], :outgoing [{:id :linear-1}]}
             {:id :feature, :incoming [{:id :linear-1}], :outgoing [{:id :feature}]}
             {:id :linear-2, :incoming [{:id :feature}], :outgoing [{:id :linear-2}]}
             {:id :output, :incoming [{:id :linear-2}], :outgoing [{:id :output}]}]
            (get inference-traversal :forward))))
    (is (= [nil nil]
           (minimal-diff
            {{:id :batch-normalization-1} {:dimension {:channels 50, :height 4, :width 4}},
             {:id :convolutional-1} {:dimension {:channels 20, :height 24, :width 24}},
             {:id :convolutional-2} {:dimension {:channels 50, :height 8, :width 8}},
             {:id :feature} {:dimension {:channels 1, :height 1, :width 500}},
             {:id :linear-1} {:dimension {:channels 1, :height 1, :width 500}},
             {:id :linear-2} {:dimension {:channels 1, :height 1, :width 10}},
             {:id :max-pooling-1} {:dimension {:channels 20, :height 12, :width 12}},
             {:id :max-pooling-2} {:dimension {:channels 50, :height 4, :width 4}},
             {:id :output} {:dimension {:channels 1, :height 1, :width 10}},
             {:id :relu-1} {:dimension {:channels 20, :height 12, :width 12}},
             {:id :relu-2} {:dimension {:channels 50, :height 4, :width 4}},
             {:stream :data} {:dimension {:channels 1, :height 28, :width 28}}}
            (get inference-traversal :buffers))))))


(deftest non-trainable-zero-attenuation
  (let [num-non-trainable 9
        src-desc (flatten mnist-description-with-toys)
        non-trainable-layers (take num-non-trainable src-desc)
        trainable-layers (drop num-non-trainable src-desc)
        new-desc (concat (map (fn [layer] (assoc layer :learning-attenuation 0))
                              non-trainable-layers)
                         trainable-layers)
        network (network/linear-network new-desc)
        traversal (traverse/training-traversal network)]
    (is (= [nil nil]
           (minimal-diff
            [{:id :output, :incoming [{:id :output}], :outgoing [{:id :linear-2}]}
             {:id :linear-2, :incoming [{:id :linear-2}], :outgoing [{:id :dropout-4}]}
             {:id :dropout-4, :incoming [{:id :dropout-4}], :outgoing [{:id :feature}]}
             {:id :feature, :incoming [{:id :feature}], :outgoing [{:id :linear-1}]}
             {:id :linear-1, :incoming [{:id :linear-1}], :outgoing [{:id :batch-normalization-1}]}
             {:id :batch-normalization-1,
              :incoming [{:id :batch-normalization-1}],
              :outgoing [{:id :dropout-3}]}]
            (get traversal :backward))))
    ;;Note that loss functions on non-trainable 'parameters' do not survive however
    ;;loss functions on non-trainable 'layers' do because they change the input gradients
    ;;for previous layers.
    (is (= #{{:type :softmax-loss,
              :output {:type :node-output, :node-id :output},
              :labels {:type :stream, :stream :output},
              :id :softmax-loss-1}
             {:type :center-loss,
              :alpha 0.9,
              :labels {:stream :output},
              :label-indexes {:stream :output},
              :label-inverse-counts {:stream :output},
              :lambda 0.05,
              :output {:type :node-output, :node-id :feature},
              :id :center-loss-1,
              :centers {:buffer-id :center-loss-1-centers-1}}}
           (traverse/gradient-loss-function network traversal)))))


(deftest non-trainable-node-non-trainable
  (let [num-non-trainable 9
        src-desc (flatten mnist-description-with-toys)
        non-trainable-layers (take num-non-trainable src-desc)
        trainable-layers (drop num-non-trainable src-desc)
        new-desc (concat (map (fn [layer] (assoc layer :non-trainable? true)) non-trainable-layers)
                         trainable-layers)
        network (network/linear-network new-desc)
        traversal (traverse/training-traversal network)]
    (is (= [nil nil]
           (minimal-diff
             [{:id :output, :incoming [{:id :output}], :outgoing [{:id :linear-2}]}
              {:id :linear-2, :incoming [{:id :linear-2}], :outgoing [{:id :dropout-4}]}
              {:id :dropout-4, :incoming [{:id :dropout-4}], :outgoing [{:id :feature}]}
              {:id :feature, :incoming [{:id :feature}], :outgoing [{:id :linear-1}]}
              {:id :linear-1, :incoming [{:id :linear-1}], :outgoing [{:id :batch-normalization-1}]}
              {:id :batch-normalization-1,
               :incoming [{:id :batch-normalization-1}],
               :outgoing [{:id :dropout-3}]}]
             (get traversal :backward))))
    (is (= #{{:type :softmax-loss,
              :output {:type :node-output, :node-id :output},
              :labels {:type :stream, :stream :output},
              :id :softmax-loss-1}
             {:type :center-loss,
              :alpha 0.9,
              :labels {:stream :output},
              :label-indexes {:stream :output},
              :label-inverse-counts {:stream :output},
              :lambda 0.05,
              :id :center-loss-1,
              :output {:type :node-output, :node-id :feature},
              :centers {:buffer-id :center-loss-1-centers-1}}}
           (traverse/gradient-loss-function network traversal)))))

(deftest appending-layers-to-network
  (testing
    "Ensures that a network built by piecing together a built-network and set of layers is
  effectively equal to building a network with a complete description"
    (let [layer-split 9
          src-desc (flatten mnist-description-with-toys)
          bottom-layers (take layer-split src-desc)
          bottom-network (network/linear-network bottom-layers)
          ;; Added io binding and traversals to make sure that when
          ;; the network is modified and rebuilt, these 2 steps are also rebuilt correctly
          top-layers (drop layer-split src-desc)
          top-network (network/assoc-layers-to-network bottom-network top-layers)
          traversal-after-stacking (traverse/training-traversal top-network)
          original-network (network/linear-network mnist-description-with-toys)
          original-traversal (traverse/training-traversal original-network)
          inference-traversal-top (traverse/inference-traversal top-network)
          inference-traversal-original (traverse/inference-traversal original-network)
          training-traversal-top (traverse/training-traversal top-network)
          training-traversal-original (traverse/training-traversal original-network)
          compute-graph->buffer-id-size-fn #(reduce (fn [m [id {:keys [buffer]}]]
                                                      (assoc m id (m/ecount buffer))) {} %)]
      (is (= [nil nil]
             (minimal-diff
              (get original-traversal :backward)
              (get traversal-after-stacking :backward))))
      (is (= [nil nil]
             (minimal-diff
              (get original-traversal :forward)
              (get traversal-after-stacking :forward))))
      (is (= [nil nil]
             (minimal-diff
              (get inference-traversal-top :buffers)
              (get inference-traversal-original :buffers))))
      (is (= [nil nil]
             (minimal-diff
              (get training-traversal-top :buffers)
              (get training-traversal-original :buffers))))
      (is (nil? (:verification-failures top-network)))
      (is (= (graph/parameter-count (network/network->graph top-network))
             (graph/parameter-count (network/network->graph original-network))))
      (is (= (compute-graph->buffer-id-size-fn (get-in top-network [:compute-graph :buffers]))
             (compute-graph->buffer-id-size-fn (get-in original-network [:compute-graph :buffers])))))))


(deftest remove-layers-from-network
  (let [mnist-net (network/linear-network mnist-description-with-toys)
        chopped-net (network/dissoc-layers-from-network mnist-net :dropout-4)]
    (is (= #{:output :linear-2 :dropout-4 :softmax-loss-1}
           (clojure.set/difference
             (set (keys (get-in mnist-net [:compute-graph :nodes])))
             (set (keys (get-in chopped-net [:compute-graph :nodes]))))))
    (is (= #{[:feature :dropout-4] [:dropout-4 :linear-2] [:linear-2 :output]
             [:output :softmax-loss-1]}
           (clojure.set/difference
             (set (get-in mnist-net [:compute-graph :edges]))
             (set (get-in chopped-net [:compute-graph :edges])))))
    (is (= #{:linear-2-bias-1 :linear-2-weights-1}
           (clojure.set/difference
             (set (keys (get-in mnist-net [:compute-graph :buffers])))
             (set (keys (get-in chopped-net [:compute-graph :buffers]))))))))



(deftest inference-after-train
  (let [network (build-big-description)]
    (is (= #{:output}
           (network/output-node-ids network :inference)))
    (is (= #{:output :feature :convolutional-2}
           (network/output-node-ids network :training)))))


(deftest concatenate-traversal-1
  (let [network (-> (network/linear-network [(layers/input 10 10 10)
                                            (layers/linear 500 :id :right)
                                            (layers/input 500 1 1 :parents [] :id :left)
                                            (layers/concatenate :parents [:left :right]
                                                                :id :concat)
                                            (layers/linear 10)]))
        train-traversal (traverse/training-traversal network)
        inference-traversal (traverse/inference-traversal network)]
    (is (= [nil nil]
           (minimal-diff
            [{:id :linear-1, :incoming [{:id :linear-1}], :outgoing [{:id :concat}]}
             {:id :concat, :incoming [{:id :concat}], :outgoing [{:stream :left} {:id :right}]}
             {:id :right, :incoming [{:id :right}], :outgoing [{:stream :input-1}]}]
            (get train-traversal :backward))))

    (is (= [nil nil]
           (minimal-diff
            {{:id :concat} {:dimension {:channels 1, :height 1, :width 1000}},
             {:id :linear-1} {:dimension {:channels 1, :height 1, :width 10}},
             {:id :right} {:dimension {:channels 1, :height 1, :width 500}},
             {:stream :input-1} {:dimension {:channels 10, :height 10, :width 10}},
             {:stream :left} {:dimension {:channels 1, :height 1, :width 500}}
             {:stream :linear-1} {:dimension {:channels 1, :height 1, :width 10}}}
            (get train-traversal :buffers))))

    (is (= [nil nil]
           (minimal-diff
            [{:id :right, :incoming [{:stream :input-1}], :outgoing [{:id :right}]}
             {:id :concat, :incoming [{:stream :left} {:id :right}], :outgoing [{:id :concat}]}
             {:id :linear-1, :incoming [{:id :concat}], :outgoing [{:id :linear-1}]}]
            (get inference-traversal :forward))))))

(deftest concatenate-traversal-2
  (let [train-traversal (-> (network/linear-network [(layers/input 10 10 10)
                                                     (layers/linear 500 :id :right)
                                                     (layers/input 500 1 1 :parents [] :id :left)
                                                     ;;Switch the left and right nodes.  Attempting to
                                                     ;;ensure we don't have some hidden dependency upon
                                                     ;;order of layer declaration.
                                                     (layers/concatenate :parents [:right :left]
                                                                         :id :concat)
                                                     (layers/linear 10)])
                            traverse/training-traversal)]
    (is (= [nil nil]
           (minimal-diff
            [{:id :linear-1, :incoming [{:id :linear-1}], :outgoing [{:id :concat}]}
             {:id :concat, :incoming [{:id :concat}], :outgoing [{:id :right} {:stream :left}]}
             {:id :right, :incoming [{:id :right}], :outgoing [{:stream :input-1}]}]
            (get train-traversal :backward))))

    (is (= [nil nil]
           (minimal-diff
            {{:id :concat} {:dimension {:channels 1, :height 1, :width 1000}},
             {:id :linear-1} {:dimension {:channels 1, :height 1, :width 10}},
             {:id :right} {:dimension {:channels 1, :height 1, :width 500}},
             {:stream :input-1} {:dimension {:channels 10, :height 10, :width 10}},
             {:stream :left} {:dimension {:channels 1, :height 1, :width 500}}
             {:stream :linear-1} {:dimension {:channels 1, :height 1, :width 10}}},
            (get train-traversal :buffers))))))


(deftest split-traversal
  (let [train-traversal (-> (network/linear-network [(layers/input 50)
                                                     (layers/split :id :split)
                                                     ;;Check for buffer id collision
                                                     (layers/split)
                                                     (layers/linear 10 :id :double-split)
                                                     (layers/linear 20
                                                                    :parents [:split]
                                                                    :id :single-split)])
                            traverse/training-traversal)]
    (is (= [nil nil]
           (minimal-diff
            [{:id :split, :incoming [{:stream :input-1}], :outgoing [{:id :split-1} {:id :split-2}]}
             {:id :split-1, :incoming [{:id :split-1}], :outgoing [{:id :split-1-1}]}
             {:id :double-split, :incoming [{:id :split-1-1}], :outgoing [{:id :double-split}]}
             {:id :single-split, :incoming [{:id :split-2}], :outgoing [{:id :single-split}]}]
            (get train-traversal :forward))))
    (is (= [nil nil]
           (minimal-diff
            {{:id :double-split} {:dimension {:channels 1, :height 1, :width 10}},
             {:id :single-split} {:dimension {:channels 1, :height 1, :width 20}},
             {:id :split-1} {:dimension {:channels 1, :height 1, :id :split-1, :width 50}},
             {:id :split-1-1} {:dimension {:channels 1, :height 1, :id :double-split, :width 50}},
             {:id :split-2} {:dimension {:channels 1, :height 1, :id :single-split, :width 50}},
             {:stream :input-1} {:dimension {:channels 1, :height 1, :width 50}}
             {:stream :single-split} {:dimension {:channels 1, :height 1, :width 20}},
             {:stream :double-split} {:dimension {:channels 1, :height 1, :width 10}}}
            (get train-traversal :buffers))))))
