(ns cortex.nn.traverse-test
  (:require [cortex.nn.traverse :as traverse]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
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
   (layers/batch-normalization 0.9)
   (layers/linear 500) ;;If you use this description put that at 1000
   (layers/relu :id :feature)
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
         :loss-function (mapv #(dissoc % :centers) (get traversal :loss-function))))


(defn minimal-diff
  [lhs rhs]
  (->> (data/diff lhs rhs)
       (take 2)
       vec))


(defn build-big-description
  []
  (let [input-bindings [(traverse/->input-binding :input-1 :data)]
        output-bindings [(traverse/->output-binding :softmax-1 :stream :labels :loss (loss/softmax-loss))
                         (traverse/->output-binding :feature
                                                    :stream :labels
                                                    :loss (loss/center-loss :alpha 0.9 :lambda 0.05))]]
    (-> (network/build-network mnist-description-with-toys)
        (traverse/bind-input-bindings input-bindings)
        (traverse/bind-output-bindings output-bindings))))

(def stream->size-map {:data 768
                       :labels 10})


(deftest big-description
  (let [network (build-big-description)
        gradient-descent (->> (traverse/network->training-traversal network stream->size-map)
                              :traversal
                              realize-traversals)
        inference-mem (->> (traverse/network->inference-traversal network stream->size-map)
                           :traversal
                           realize-traversals)]
    (is (= 434280 (get network :parameter-count)))
    (is (= 434280 (->> (get-in network [:layer-graph :buffers])
                       (map (comp m/ecount :buffer second))
                       (reduce +))))
    (is (= [nil nil]
           (minimal-diff
            [{:id :dropout-1, :incoming [{:input-stream :data}], :outgoing [{:id :dropout-1}]}
             {:id :convolutional-1, :incoming [{:id :dropout-1}], :outgoing [{:id :convolutional-1}]}
             {:id :max-pooling-1, :incoming [{:id :convolutional-1}], :outgoing [{:id :max-pooling-1}]}
             {:id :relu-1, :incoming [{:id :max-pooling-1}], :outgoing [{:id :relu-1}]}
             {:id :dropout-2, :incoming [{:id :relu-1}], :outgoing [{:id :dropout-2}]}
             {:id :convolutional-2, :incoming [{:id :dropout-2}], :outgoing [{:id :convolutional-2}]}
             {:id :max-pooling-2, :incoming [{:id :convolutional-2}], :outgoing [{:id :max-pooling-2}]}
             {:id :relu-2, :incoming [{:id :max-pooling-2}], :outgoing [{:id :relu-2}]}
             {:id :dropout-3, :incoming [{:id :relu-2}], :outgoing [{:id :dropout-3}]}
             {:id :batch-normalization-1, :incoming [{:id :dropout-3}], :outgoing [{:id :batch-normalization-1}]}
             {:id :linear-1, :incoming [{:id :batch-normalization-1}], :outgoing [{:id :linear-1}]}
             {:id :feature, :incoming [{:id :linear-1}], :outgoing [{:output-id :feature}]}
             {:id :dropout-4, :incoming [{:output-id :feature}], :outgoing [{:id :dropout-4}]}
             {:id :linear-2, :incoming [{:id :dropout-4}], :outgoing [{:id :linear-2}]}
             {:id :softmax-1, :incoming [{:id :linear-2}], :outgoing [{:output-id :softmax-1}]}]
            (get gradient-descent :forward))))
    (is (= [nil nil]
           (minimal-diff
            {{:id :batch-normalization-1} {:id :batch-normalization-1, :size 800},
             {:id :convolutional-1} {:id :convolutional-1, :size 11520},
             {:id :convolutional-2} {:id :convolutional-2, :size 3200},
             {:id :dropout-1} {:id :dropout-1, :size 784},
             {:id :dropout-2} {:id :dropout-2, :size 2880},
             {:id :dropout-3} {:id :dropout-3, :size 800},
             {:id :dropout-4} {:id :dropout-4, :size 500},
             {:id :linear-1} {:id :linear-1, :size 500},
             {:id :linear-2} {:id :linear-2, :size 10},
             {:id :max-pooling-1} {:id :max-pooling-1, :size 2880},
             {:id :max-pooling-2} {:id :max-pooling-2, :size 800},
             {:id :relu-1} {:id :relu-1, :size 2880},
             {:id :relu-2} {:id :relu-2, :size 800},
             {:input-stream :data} {:input-stream :data, :size 784},
             {:output-id :feature} {:loss {:alpha 0.9, :lambda 0.05, :type :center-loss},
                                    :output-id :feature,
                                    :output-stream :labels,
                                    :size 500},
             {:output-id :softmax-1} {:loss {:type :softmax-loss},
                                      :output-id :softmax-1,
                                      :output-stream :labels,
                                      :size 10}}
            (get gradient-descent :buffers))))
    (is (= [nil nil]
           (minimal-diff
            [{:type :softmax-loss,
              :output {:data {:type :node-output, :node-id :softmax-1}},
              :labels {:data {:type :stream, :stream :labels}}}
             {:type :center-loss,
              :alpha 0.9,
              :lambda 0.05,
              :output {:data {:type :node-output, :node-id :feature}},
              :labels {:data {:type :stream, :stream :labels}}}
             {:type :l2-regularization,
              :lambda 0.01,
              :output {:data {:type :node-output, :node-id :convolutional-2}}}
             {:type :l1-regularization,
              :lambda 0.001,
              :output
              {:data
               {:type :node-parameter,
                :node-id :convolutional-1,
                :parameter :weights}}}]
            (get gradient-descent :loss-function))))
    (is (= [nil nil]
           (minimal-diff
            [{:id :convolutional-1, :incoming [{:input-stream :data}], :outgoing [{:id :convolutional-1}]}
             {:id :max-pooling-1, :incoming [{:id :convolutional-1}], :outgoing [{:id :max-pooling-1}]}
             {:id :relu-1, :incoming [{:id :max-pooling-1}], :outgoing [{:id :relu-1}]}
             {:id :convolutional-2, :incoming [{:id :relu-1}], :outgoing [{:id :convolutional-2}]}
             {:id :max-pooling-2, :incoming [{:id :convolutional-2}], :outgoing [{:id :max-pooling-2}]}
             {:id :relu-2, :incoming [{:id :max-pooling-2}], :outgoing [{:id :relu-2}]}
             {:id :batch-normalization-1, :incoming [{:id :relu-2}], :outgoing [{:id :batch-normalization-1}]}
             {:id :linear-1, :incoming [{:id :batch-normalization-1}], :outgoing [{:id :linear-1}]}
             {:id :feature, :incoming [{:id :linear-1}], :outgoing [{:output-id :feature}]}
             {:id :linear-2, :incoming [{:output-id :feature}], :outgoing [{:id :linear-2}]}
             {:id :softmax-1, :incoming [{:id :linear-2}], :outgoing [{:output-id :softmax-1}]}]
            (get inference-mem :forward))))
    (is (= [nil nil]
           (minimal-diff
            {{:id :batch-normalization-1} {:id :batch-normalization-1, :size 800},
             {:id :convolutional-1} {:id :convolutional-1, :size 11520},
             {:id :convolutional-2} {:id :convolutional-2, :size 3200},
             {:id :linear-1} {:id :linear-1, :size 500},
             {:id :linear-2} {:id :linear-2, :size 10},
             {:id :max-pooling-1} {:id :max-pooling-1, :size 2880},
             {:id :max-pooling-2} {:id :max-pooling-2, :size 800},
             {:id :relu-1} {:id :relu-1, :size 2880},
             {:id :relu-2} {:id :relu-2, :size 800},
             {:input-stream :data} {:input-stream :data, :size 784},
             {:output-id :feature} {:loss {:alpha 0.9, :lambda 0.05, :type :center-loss},
                                    :output-id :feature,
                                    :output-stream :labels,
                                    :size 500},
             {:output-id :softmax-1} {:loss {:type :softmax-loss},
                                      :output-id :softmax-1,
                                      :output-stream :labels,
                                      :size 10}}
            (get inference-mem :buffers))))))

(def test-data (atom nil))


(deftest non-trainable-zero-attenuation
  (let [num-non-trainable 9
        src-desc (flatten mnist-description-with-toys)
        non-trainable-layers (take num-non-trainable src-desc)
        trainable-layers (drop num-non-trainable src-desc)
        new-desc (concat (map (fn [layer] (assoc layer :learning-attenuation 0)) non-trainable-layers)
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
            [{:labels {:data {:stream :labels, :type :stream}},
              :output {:data {:node-id :softmax-1, :type :node-output}},
              :type :softmax-loss}]
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
            [{:labels {:data {:stream :labels, :type :stream}},
              :output {:data {:node-id :softmax-1, :type :node-output}},
              :type :softmax-loss}]
            (get traversal :loss-function))))))
