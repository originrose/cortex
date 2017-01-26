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
   (layers/convolutional 5 0 1 20)
   (layers/max-pooling 2 0 2)
   (layers/relu)
   (layers/dropout 0.75)
   (layers/convolutional 5 0 1 50)
   (layers/max-pooling 2 0 2)
   (layers/relu)
   (layers/dropout 0.75)
   (layers/batch-normalization 0.9)
   (layers/linear->relu 500) ;;If you use this description put that at 1000
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
         :backward (realize-traversal-pass backward)))


(defn minimal-diff
  [lhs rhs]
  (->> (data/diff lhs rhs)
       (take 2)
       vec))


(deftest build-big-description
  (let [input-bindings [(traverse/->input-binding :input-1 :data)]
        output-bindings [(traverse/->output-binding :softmax-1 :stream :labels :loss (loss/softmax-loss))]
        network (-> (network/build-network mnist-description-with-toys)
                    (traverse/bind-input-bindings input-bindings)
                    (traverse/bind-output-bindings output-bindings))
        gradient-descent (->> (traverse/network->training-traversal network)
                              :traversal
                              realize-traversals)
        inference-mem (->> (traverse/network->inference-traversal network)
                           :traversal
                           realize-traversals)]
    (is (= 434280 (get network :parameter-count)))
    (is (= 434280 (->> (get-in network [:layer-graph :buffers])
                       (map (comp m/ecount :buffer :buffer second))
                       (reduce +))))
    (is (= [nil nil]
           (minimal-diff
            [{:id :dropout-1, :incoming [{:input-stream :data}], :outgoing [{:id :convolutional-1}]}
             {:id :convolutional-1, :incoming [{:id :convolutional-1}], :outgoing [{:id :max-pooling-1}]}
             {:id :max-pooling-1, :incoming [{:id :max-pooling-1}], :outgoing [{:id :relu-1}]}
             {:id :relu-1, :incoming [{:id :relu-1}], :outgoing [{:id :dropout-2}]}
             {:id :dropout-2, :incoming [{:id :dropout-2}], :outgoing [{:id :convolutional-2}]}
             {:id :convolutional-2, :incoming [{:id :convolutional-2}], :outgoing [{:id :max-pooling-2}]}
             {:id :max-pooling-2, :incoming [{:id :max-pooling-2}], :outgoing [{:id :relu-2}]}
             {:id :relu-2, :incoming [{:id :relu-2}], :outgoing [{:id :dropout-3}]}
             {:id :dropout-3, :incoming [{:id :dropout-3}], :outgoing [{:id :batch-normalization-1}]}
             {:id :batch-normalization-1, :incoming [{:id :batch-normalization-1}], :outgoing [{:id :linear-1}]}
             {:id :linear-1, :incoming [{:id :linear-1}], :outgoing [{:id :relu-3}]}
             {:id :relu-3, :incoming [{:id :relu-3}], :outgoing [{:id :dropout-4}]}
             {:id :dropout-4, :incoming [{:id :dropout-4}], :outgoing [{:id :linear-2}]}
             {:id :linear-2, :incoming [{:id :linear-2}], :outgoing [{:id :softmax-1}]}
             {:id :softmax-1,
              :incoming [{:id :softmax-1}],
              :outgoing [{:output-id :softmax-1}]}]
            (get gradient-descent :forward))))
    (is (= [nil nil]
           (minimal-diff
            {{:id :max-pooling-2} {:id :max-pooling-2, :size 3200},
             {:id :convolutional-1} {:id :convolutional-1, :size 784},
             {:id :batch-normalization-1}
             {:id :batch-normalization-1, :size 800},
             {:id :relu-2} {:id :relu-2, :size 800},
             {:id :dropout-3} {:id :dropout-3, :size 800},
             {:id :linear-2} {:id :linear-2, :size 500},
             {:id :softmax-1} {:id :softmax-1, :size 10},
             {:id :relu-3} {:id :relu-3, :size 500},
             {:input-stream :data} {:input-stream :data, :size 784},
             {:id :dropout-4} {:id :dropout-4, :size 500},
             {:id :max-pooling-1} {:id :max-pooling-1, :size 11520},
             {:id :linear-1} {:id :linear-1, :size 800},
             {:id :relu-1} {:id :relu-1, :size 2880},
             {:id :dropout-2} {:id :dropout-2, :size 2880},
             {:output-id :softmax-1}
             {:output-id :softmax-1,
              :output-stream :labels,
              :loss {:type :softmax-loss }
              :size 10},
             {:id :convolutional-2} {:id :convolutional-2, :size 2880}}
            (get gradient-descent :buffers))))
    (is (= [nil nil]
           (minimal-diff
            [{:id :convolutional-1, :incoming [{:input-stream :data}], :outgoing [{:id :max-pooling-1}]}
             {:id :max-pooling-1, :incoming [{:id :max-pooling-1}], :outgoing [{:id :relu-1}]}
             {:id :relu-1, :incoming [{:id :relu-1}], :outgoing [{:id :convolutional-2}]}
             {:id :convolutional-2, :incoming [{:id :convolutional-2}], :outgoing [{:id :max-pooling-2}]}
             {:id :max-pooling-2, :incoming [{:id :max-pooling-2}], :outgoing [{:id :relu-2}]}
             {:id :relu-2, :incoming [{:id :relu-2}], :outgoing [{:id :batch-normalization-1}]}
             {:id :batch-normalization-1, :incoming [{:id :batch-normalization-1}], :outgoing [{:id :linear-1}]}
             {:id :linear-1, :incoming [{:id :linear-1}], :outgoing [{:id :relu-3}]}
             {:id :relu-3, :incoming [{:id :relu-3}], :outgoing [{:id :linear-2}]}
             {:id :linear-2, :incoming [{:id :linear-2}], :outgoing [{:id :softmax-1}]}
             {:id :softmax-1,
              :incoming [{:id :softmax-1}],
              :outgoing [{:output-id :softmax-1}]}]
            (get inference-mem :forward))))
    (is (= {{:id :max-pooling-2} {:id :max-pooling-2, :size 3200},
            {:id :batch-normalization-1}
            {:id :batch-normalization-1, :size 800},
            {:id :relu-2} {:id :relu-2, :size 800},
            {:id :linear-2} {:id :linear-2, :size 500},
            {:id :softmax-1} {:id :softmax-1, :size 10},
            {:id :relu-3} {:id :relu-3, :size 500},
            {:input-stream :data} {:input-stream :data, :size 784},
            {:id :max-pooling-1} {:id :max-pooling-1, :size 11520},
            {:id :linear-1} {:id :linear-1, :size 800},
            {:id :relu-1} {:id :relu-1, :size 2880},
            {:output-id :softmax-1}
            {:output-id :softmax-1,
             :output-stream :labels,
             :loss {:type :softmax-loss}
             :size 10},
            {:id :convolutional-2} {:id :convolutional-2, :size 2880}}
           (get inference-mem :buffers)))))


(deftest non-trainable-zero-attenuation
  (let [num-non-trainable 9
        src-desc (flatten mnist-description-with-toys)
        non-trainable-layers (take num-non-trainable src-desc)
        trainable-layers (drop num-non-trainable src-desc)
        new-desc (concat (map (fn [layer] (assoc layer :learning-attenuation 0)) non-trainable-layers)
                         trainable-layers)
        network (-> (network/build-network new-desc)
                    traverse/auto-bind-io
                    traverse/network->training-traversal)
        traversal (-> (get network :traversal)
                      realize-traversals)]
    (is (= [nil nil]
           (minimal-diff
            [{:id :softmax-1, :incoming [{:output-id :softmax-1}], :outgoing [{:id :softmax-1}]}
             {:id :linear-2, :incoming [{:id :softmax-1}], :outgoing [{:id :linear-2}]}
             {:id :dropout-4, :incoming [{:id :linear-2}], :outgoing [{:id :dropout-4}]}
             {:id :relu-3, :incoming [{:id :dropout-4}], :outgoing [{:id :relu-3}]}
             {:id :linear-1, :incoming [{:id :relu-3}], :outgoing [{:id :linear-1}]}
             {:id :batch-normalization-1, :incoming [{:id :linear-1}], :outgoing [{:id :batch-normalization-1}]}]
            (get traversal :backward))))))


(deftest non-trainable-node-non-trainable
  (let [num-non-trainable 9
        src-desc (flatten mnist-description-with-toys)
        non-trainable-layers (take num-non-trainable src-desc)
        trainable-layers (drop num-non-trainable src-desc)
        new-desc (concat (map (fn [layer] (assoc layer :non-trainable? true)) non-trainable-layers)
                         trainable-layers)
        network (-> (network/build-network new-desc)
                    traverse/auto-bind-io
                    traverse/network->training-traversal)
        traversal (-> (get network :traversal)
                      realize-traversals)]
    (is (= [nil nil]
           (minimal-diff
            [{:id :softmax-1, :incoming [{:output-id :softmax-1}], :outgoing [{:id :softmax-1}]}
             {:id :linear-2, :incoming [{:id :softmax-1}], :outgoing [{:id :linear-2}]}
             {:id :dropout-4, :incoming [{:id :linear-2}], :outgoing [{:id :dropout-4}]}
             {:id :relu-3, :incoming [{:id :dropout-4}], :outgoing [{:id :relu-3}]}
             {:id :linear-1, :incoming [{:id :relu-3}], :outgoing [{:id :linear-1}]}
             {:id :batch-normalization-1, :incoming [{:id :linear-1}], :outgoing [{:id :batch-normalization-1}]}]
            (get traversal :backward))))))

(deftest appending-layers-to-network
  "This test ensures that a network built by piecing together a built-network
  and set of layers is effectively equal to building a network with a complete description"
  (let [layer-split 9
        src-desc (flatten mnist-description-with-toys)
        bottom-layers (take layer-split src-desc)
        bottom-network (-> (network/build-network bottom-layers)
                         traverse/auto-bind-io
                         traverse/network->training-traversal)
        ;; Added io binding and traversals to make sure that when
        ;; the network is modified and rebuilt, these 2 steps are also rebuilt correctly

        top-layers (drop layer-split src-desc)
        top-network-desc (network/assoc-layers-to-network bottom-network top-layers)
        top-network (-> (network/build-network top-network-desc)
                      traverse/auto-bind-io
                      traverse/network->training-traversal)

        traversal-after-stacking (-> (get top-network :traversal)
                    realize-traversals)

        original-network (-> (network/build-network mnist-description-with-toys)
                  traverse/auto-bind-io
                  traverse/network->training-traversal)
        original-traversal (-> (get original-network :traversal)
                             realize-traversals)

        inference-mem-top (->> (traverse/network->inference-traversal top-network)
                        :traversal
                        realize-traversals)

        inference-mem-original (->> (traverse/network->inference-traversal original-network)
                        :traversal
                        realize-traversals)

        gradient-descent-top (->> (traverse/network->training-traversal top-network)
                           :traversal
                           realize-traversals)

        gradient-descent-original (->> (traverse/network->training-traversal original-network)
                           :traversal
                           realize-traversals)]
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
    (is (= (:parameter-count top-network) (:parameter-count original-network)))))


(deftest remove-layers-from-network
  (let [mnist-net (-> (network/build-network mnist-description-with-toys)
                  traverse/auto-bind-io
                  traverse/network->training-traversal)
        chopped-net (network/dissoc-layers-from-network mnist-net :relu-3)]
    (is (= #{:softmax-1 :linear-2 :dropout-4 :relu-3}
           (clojure.set/difference
             (set (keys (get-in mnist-net [:layer-graph :id->node-map])))
             (set (keys (get-in chopped-net [:layer-graph :id->node-map]))))))
    (is (= #{[:relu-3 :dropout-4] [:dropout-4 :linear-2] [:linear-2 :softmax-1] [:linear-1 :relu-3]}
           (clojure.set/difference
             (set (get-in mnist-net [:layer-graph :edges]))
             (set (get-in chopped-net [:layer-graph :edges])))))
    (is (= #{:linear-2-bias :linear-2-weights}
           (clojure.set/difference
             (set (keys (get-in mnist-net [:layer-graph :buffers])))
             (set (keys (get-in chopped-net [:layer-graph :buffers]))))))))
