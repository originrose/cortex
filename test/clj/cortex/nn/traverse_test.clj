(ns cortex.nn.traverse-test
  (:require [cortex.nn.traverse :as traverse]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
            [cortex.graph :as graph]
            [clojure.test :refer :all]
            [clojure.core.matrix :as m]
            [cortex.loss :as loss]
            [clojure.data :as data]))


(def mnist-description-with-toys
  [(layers/input 28 28 1)
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
   (layers/relu :id :feature :center-loss {:labels {:stream :labels}
                                           :label-indexes {:stream :labels}
                                           :label-inverse-counts {:stream :labels}
                                           :lambda 0.05
                                           :alpha 0.9})
   (layers/dropout 0.5)
   (layers/linear->softmax 10)])


(defn realize-traversal-pass
  [pass]
  (mapv (fn [{:keys [incoming outgoing] :as item}]
          (assoc item
                 :incoming (if (seq? incoming)
                             (vec incoming)
                             incoming)
                 :outgoing (if (seq? outgoing)
                             (vec outgoing)
                             outgoing)))
        pass))


(defn realize-traversals
  "Make sure the traversals are actual vectors and not lazy sequences"
  [{:keys [forward backward] :as traversal}]
  (assoc traversal
         :forward (realize-traversal-pass forward)
         :backward (realize-traversal-pass backward)
         :loss-function (vec (get traversal :loss-function))))


(defn minimal-diff
  [lhs rhs]
  (->> (data/diff lhs rhs)
       (take 2)
       vec))


(defn build-big-description
  []
  (let [input-bindings [(traverse/->input-binding :input-1 :data)]
        output-bindings [(traverse/->output-binding :softmax-1 :stream
                                                    :labels :loss (loss/softmax-loss))]]
    (-> (network/build-network mnist-description-with-toys)
        (traverse/bind-input-bindings input-bindings)
        (traverse/bind-output-bindings output-bindings))))



(def stream->size-map {:data 768
                       :labels 10})


(deftest big-description
  (let [network (build-big-description)
        ;;Run the traversal twice as it is supposed to be idempotent.
        training-net (-> (traverse/network->training-traversal network stream->size-map)
                         (traverse/network->training-traversal stream->size-map))
        gradient-descent (->> training-net
                              :traversal
                              realize-traversals)
        inference-mem (->> (traverse/network->inference-traversal network stream->size-map)
                           :traversal
                           realize-traversals)]
    (is (= 434280 (graph/parameter-count (get network :layer-graph))))
    (is (= 434280 (->> (get-in network [:layer-graph :buffers])
                       (map (comp m/ecount :buffer second))
                       (reduce +))))
    ;;Adding in the parameters required for the center loss centers.  10 * 500 = 5000
    ;;extra parameters
    (is (= 439280 (graph/parameter-count (get training-net :layer-graph))))
    (is (= :node-argument (-> (network/network->graph training-net)
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
             {:id :softmax-1, :incoming [{:id :linear-2}], :outgoing [{:output-id :softmax-1}]}]
            (get gradient-descent :forward))))
    (is (= [nil nil]
           (minimal-diff
            {{:id :max-pooling-2}
             {:id :max-pooling-2,
              :dimension {:channels 50, :height 4, :width 4},},
             {:id :convolutional-1}
             {:id :convolutional-1,
              :dimension {:channels 20, :height 24, :width 24},},
             {:id :batch-normalization-1}
             {:id :batch-normalization-1,
              :dimension {:channels 50, :height 4, :width 4},},
             {:id :relu-2}
             {:id :relu-2,
              :dimension {:channels 50, :height 4, :width 4},},
             {:id :dropout-3}
             {:id :dropout-3,
              :dimension {:channels 50, :height 4, :width 4},},
             {:stream :data}
             {:stream :data,
              :dimension {:channels 1, :height 28, :width 28},},
             {:id :linear-2}
             {:id :linear-2,
              :dimension {:channels 1, :height 1, :width 10},},
             {:id :dropout-4}
             {:id :dropout-4,
              :dimension {:channels 1, :height 1, :width 500},},
             {:id :max-pooling-1}
             {:id :max-pooling-1,
              :dimension {:channels 20, :height 12, :width 12},},
             {:id :linear-1}
             {:id :linear-1,
              :dimension {:channels 1, :height 1, :width 500},},
             {:id :relu-1}
             {:id :relu-1,
              :dimension {:channels 20, :height 12, :width 12},},
             {:id :feature}
             {:id :feature,
              :dimension {:channels 1, :height 1, :width 500},},
             {:id :dropout-2}
             {:id :dropout-2,
              :dimension {:channels 20, :height 12, :width 12},},
             {:output-id :softmax-1}
             {:output-id :softmax-1,
              :loss {:type :softmax-loss},
              :dimension {:channels 1, :height 1, :width 10},},
             {:id :convolutional-2}
             {:id :convolutional-2,
              :dimension {:channels 50, :height 8, :width 8},},
             {:id :dropout-1}
             {:id :dropout-1,
              :dimension {:channels 1, :height 28, :width 28},}}
            (get gradient-descent :buffers))))
    (is (= [nil nil]
           (minimal-diff
            [{:type :softmax-loss,
              :output {:type :node-output, :node-id :softmax-1},
              :labels {:type :stream, :stream :labels},
              :id :softmax-loss-1}
             {:type :center-loss,
              :alpha 0.9,
              :labels {:stream :labels},
              :label-indexes {:stream :labels},
              :label-inverse-counts {:stream :labels},
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
              :id :l1-regularization-1}]
            (get gradient-descent :loss-function))))

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
             {:id :softmax-1, :incoming [{:id :linear-2}], :outgoing [{:output-id :softmax-1}]}]
            (get inference-mem :forward))))
    (is (= [nil nil]
           (minimal-diff
            {{:id :max-pooling-2}
             {:id :max-pooling-2,
              :dimension {:channels 50, :height 4, :width 4},},
             {:id :convolutional-1}
             {:id :convolutional-1,
              :dimension {:channels 20, :height 24, :width 24},},
             {:id :batch-normalization-1}
             {:id :batch-normalization-1,
              :dimension {:channels 50, :height 4, :width 4},},
             {:id :relu-2}
             {:id :relu-2,
              :dimension {:channels 50, :height 4, :width 4},},
             {:stream :data}
             {:stream :data,
              :dimension {:channels 1, :height 28, :width 28},},
             {:id :linear-2}
             {:id :linear-2,
              :dimension {:channels 1, :height 1, :width 10},},
             {:id :max-pooling-1}
             {:id :max-pooling-1,
              :dimension {:channels 20, :height 12, :width 12},},
             {:id :linear-1}
             {:id :linear-1,
              :dimension {:channels 1, :height 1, :width 500},},
             {:id :relu-1}
             {:id :relu-1,
              :dimension {:channels 20, :height 12, :width 12},},
             {:id :feature}
             {:id :feature,
              :dimension {:channels 1, :height 1, :width 500},},
             {:output-id :softmax-1}
             {:output-id :softmax-1,
              :loss {:type :softmax-loss},
              :dimension {:channels 1, :height 1, :width 10},},
             {:id :convolutional-2}
             {:id :convolutional-2,
              :dimension {:channels 50, :height 8, :width 8},}}
            (get inference-mem :buffers))))))

(def test-data (atom nil))


(deftest non-trainable-zero-attenuation
  (let [num-non-trainable 9
        src-desc (flatten mnist-description-with-toys)
        non-trainable-layers (take num-non-trainable src-desc)
        trainable-layers (drop num-non-trainable src-desc)
        new-desc (concat (map (fn [layer] (assoc layer :learning-attenuation 0))
                              non-trainable-layers)
                         trainable-layers)
        network (-> (network/build-network new-desc)
                    traverse/auto-bind-io
                    (traverse/network->training-traversal stream->size-map))
        traversal (-> (get network :traversal)
                      realize-traversals)]
    (reset! test-data traversal)
    (is (= [nil nil]
           (minimal-diff
            [{:id :softmax-1, :incoming [{:output-id :softmax-1}], :outgoing [{:id :linear-2}]}
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
    (is (= [nil nil]
           (minimal-diff
            [{:type :softmax-loss,
              :output {:type :node-output, :node-id :softmax-1},
              :labels {:type :stream, :stream :labels},
              :id :softmax-loss-1}
             {:type :center-loss,
              :alpha 0.9,
              :labels {:stream :labels},
              :label-indexes {:stream :labels},
              :label-inverse-counts {:stream :labels},
              :lambda 0.05,
              :output {:type :node-output, :node-id :feature},
              :id :center-loss-1,
              :centers {:buffer-id :center-loss-1-centers-1}}]
            (get traversal :loss-function))))))


(deftest non-trainable-node-non-trainable
  (let [num-non-trainable 9
        src-desc (flatten mnist-description-with-toys)
        non-trainable-layers (take num-non-trainable src-desc)
        trainable-layers (drop num-non-trainable src-desc)
        new-desc (concat (map (fn [layer] (assoc layer :non-trainable? true)) non-trainable-layers)
                         trainable-layers)
        network (-> (network/build-network new-desc)
                    traverse/auto-bind-io
                    (traverse/network->training-traversal stream->size-map))
        traversal (-> (get network :traversal)
                      realize-traversals)]
    (is (= [nil nil]
           (minimal-diff
             [{:id :softmax-1, :incoming [{:output-id :softmax-1}], :outgoing [{:id :linear-2}]}
              {:id :linear-2, :incoming [{:id :linear-2}], :outgoing [{:id :dropout-4}]}
              {:id :dropout-4, :incoming [{:id :dropout-4}], :outgoing [{:id :feature}]}
              {:id :feature, :incoming [{:id :feature}], :outgoing [{:id :linear-1}]}
              {:id :linear-1, :incoming [{:id :linear-1}], :outgoing [{:id :batch-normalization-1}]}
              {:id :batch-normalization-1,
               :incoming [{:id :batch-normalization-1}],
               :outgoing [{:id :dropout-3}]}]
             (get traversal :backward))))
    (is (= [nil nil]
           (minimal-diff
            [{:type :softmax-loss,
              :output {:type :node-output, :node-id :softmax-1},
              :labels {:type :stream, :stream :labels},
              :id :softmax-loss-1}
             {:type :center-loss,
              :alpha 0.9,
              :labels {:stream :labels},
              :label-indexes {:stream :labels},
              :label-inverse-counts {:stream :labels},
              :lambda 0.05,
              :id :center-loss-1,
              :output {:type :node-output, :node-id :feature},
              :centers {:buffer-id :center-loss-1-centers-1}}]
             (get traversal :loss-function))))))

(deftest appending-layers-to-network
  "This test ensures that a network built by piecing together a built-network
  and set of layers is effectively equal to building a network with a complete description"
  (let [layer-split 9
        src-desc (flatten mnist-description-with-toys)
        bottom-layers (take layer-split src-desc)
        bottom-network (-> (network/build-network bottom-layers)
                         traverse/auto-bind-io)
        ;; Added io binding and traversals to make sure that when
        ;; the network is modified and rebuilt, these 2 steps are also rebuilt correctly

        top-layers (drop layer-split src-desc)
        top-network (-> (network/assoc-layers-to-network bottom-network top-layers)
                        traverse/auto-bind-io
                        (traverse/network->training-traversal stream->size-map))

        traversal-after-stacking (-> (get top-network :traversal)
                                     realize-traversals)

        original-network (-> (network/build-network mnist-description-with-toys)
                             traverse/auto-bind-io
                             (traverse/network->training-traversal stream->size-map))

        original-traversal (-> (get original-network :traversal)
                               realize-traversals)

        inference-mem-top (->> (traverse/network->inference-traversal top-network stream->size-map)
                               :traversal
                               realize-traversals)

        inference-mem-original (->> (traverse/network->inference-traversal original-network stream->size-map)
                                    :traversal
                                    realize-traversals)

        gradient-descent-top (->> (traverse/network->training-traversal top-network stream->size-map)
                                  :traversal
                                  realize-traversals)

        gradient-descent-original (->> (traverse/network->training-traversal original-network stream->size-map)
                                       :traversal
                                       realize-traversals)
        layer-graph->buffer-id-size-fn #(reduce (fn [m [id {:keys [buffer]}]] (assoc m id (m/ecount buffer))) {} %)]
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
             (get inference-mem-top :buffers)
             (get inference-mem-original :buffers))))
    (is (= [nil nil]
           (minimal-diff
             (get gradient-descent-top :buffers)
             (get gradient-descent-original :buffers))))
    (is (nil? (:verification-failures top-network)))
    (is (= (graph/parameter-count (network/network->graph top-network))
           (graph/parameter-count (network/network->graph original-network))))
    (is (= (layer-graph->buffer-id-size-fn (get-in top-network [:layer-graph :buffers]))
           (layer-graph->buffer-id-size-fn (get-in original-network [:layer-graph :buffers]))))))


(deftest remove-layers-from-network
  (let [mnist-net (-> (network/build-network mnist-description-with-toys)
                    traverse/auto-bind-io)
        chopped-net (network/dissoc-layers-from-network mnist-net :dropout-4)]
    (is (= #{:softmax-1 :linear-2 :dropout-4}
           (clojure.set/difference
             (set (keys (get-in mnist-net [:layer-graph :id->node-map])))
             (set (keys (get-in chopped-net [:layer-graph :id->node-map]))))))
    (is (= #{[:feature :dropout-4] [:dropout-4 :linear-2] [:linear-2 :softmax-1]}
           (clojure.set/difference
             (set (get-in mnist-net [:layer-graph :edges]))
             (set (get-in chopped-net [:layer-graph :edges])))))
    (is (= #{:linear-2-bias-1 :linear-2-weights-1}
           (clojure.set/difference
             (set (keys (get-in mnist-net [:layer-graph :buffers])))
             (set (keys (get-in chopped-net [:layer-graph :buffers]))))))))



(deftest inference-after-train
  (let [network (build-big-description)
        training-net (traverse/network->training-traversal network stream->size-map)
        inference-net (traverse/network->inference-traversal
                       (traverse/auto-bind-io training-net) stream->size-map)
        output-bindings (vec (traverse/get-output-bindings inference-net))]
    (is (= 1 (count output-bindings)))
    (is (= :softmax-1 (get-in output-bindings [0 :node-id])))))


(deftest concatenate-traversal-1
  (let [network (-> (network/build-network [(layers/input 10 10 10)
                                            (layers/linear 500 :id :right)
                                            (layers/input 500 1 1 :parents [] :id :left)
                                            (layers/concatenate :parents [:left :right]
                                                                :id :concat)
                                            (layers/linear 10)])
                    (traverse/auto-bind-io))
        stream->size-map {:data-1 (* 25 25 10)
                          :data-2 500
                          :labels 10}
        train-network (traverse/network->training-traversal network stream->size-map)
        inference-network (traverse/network->inference-traversal network stream->size-map)
        train-traversal (-> (get train-network :traversal)
                            realize-traversals)
        inference-traversal (-> (get inference-network :traversal)
                                realize-traversals)]
    (is (= [nil nil]
           (minimal-diff
            [{:incoming [{:output-id :linear-1}],
              :id :linear-1,
              :outgoing [{:id :concat}]}
             {:incoming [{:id :concat}],
              :id :concat,
              :outgoing [{:stream :data-2} {:id :right}]}
             {:incoming [{:id :right}],
              :id :right,
              :outgoing [{:stream :data-1}]}]
            (get train-traversal :backward))))

    (is (= [nil nil]
           (minimal-diff
            {{:id :right}
             {:id :right, :dimension {:channels 1, :height 1, :width 500}},
             {:stream :data-1}
             {:stream :data-1, :dimension {:channels 10, :height 10, :width 10}},
             {:id :concat}
             {:id :concat, :dimension {:channels 1, :height 1, :width 1000}},
             {:stream :data-2}
             {:stream :data-2, :dimension {:channels 1, :height 1, :width 500}},
             {:output-id :linear-1}
             {:output-id :linear-1,
              :loss {:type :mse-loss},
              :dimension {:channels 1, :height 1, :width 10}}},
            (get train-traversal :buffers))))

    (is (= [nil nil]
           (minimal-diff
            [{:incoming [{:stream :data-1}], :id :right, :outgoing [{:id :right}]}
             {:incoming [{:stream :data-2} {:id :right}],
              :id :concat,
              :outgoing [{:id :concat}]}
             {:incoming [{:id :concat}],
              :id :linear-1,
              :outgoing [{:output-id :linear-1}]}]
            (get inference-traversal :forward))))))

(deftest concatenate-traversal-2
  (let [network (-> (network/build-network [(layers/input 10 10 10)
                                            (layers/linear 500 :id :right)
                                            (layers/input 500 1 1 :parents [] :id :left)
                                            ;;Switch the left and right nodes.  Attempting to
                                            ;;ensure we don't have some hidden dependency upon
                                            ;;order of layer declaration.
                                            (layers/concatenate :parents [:right :left]
                                                                :id :concat)
                                            (layers/linear 10)])
                    (traverse/auto-bind-io))
        stream->size-map {:data-1 (* 25 25 10)
                          :data-2 500
                          :labels 10}
        train-network (traverse/network->training-traversal network stream->size-map)
        train-traversal (-> (get train-network :traversal)
                            realize-traversals)]
    (is (= [nil nil]
           (minimal-diff
            [{:incoming [{:output-id :linear-1}],
              :id :linear-1,
              :outgoing [{:id :concat}]}
             {:incoming [{:id :concat}],
              :id :concat,
              :outgoing [{:id :right} {:stream :data-2}]}
             {:incoming [{:id :right}],
              :id :right,
              :outgoing [{:stream :data-1}]}]
            (get train-traversal :backward))))

    (is (= [nil nil]
           (minimal-diff
            {{:id :right}
             {:id :right, :dimension {:channels 1, :height 1, :width 500}},
             {:stream :data-1}
             {:stream :data-1, :dimension {:channels 10, :height 10, :width 10}},
             {:id :concat}
             {:id :concat, :dimension {:channels 1, :height 1, :width 1000}},
             {:stream :data-2}
             {:stream :data-2, :dimension {:channels 1, :height 1, :width 500}},
             {:output-id :linear-1}
             {:output-id :linear-1,
              :loss {:type :mse-loss},
              :dimension {:channels 1, :height 1, :width 10}}},
            (get train-traversal :buffers))))))
