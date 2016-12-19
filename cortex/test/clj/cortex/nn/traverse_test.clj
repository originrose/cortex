(ns cortex.nn.traverse-test
  (:require [cortex.nn.traverse :as traverse]
            [cortex.nn.layers :as layers]
            [cortex.nn.build :as build]
            [clojure.test :refer :all]
            [clojure.core.matrix :as m]
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
  (let [built-network (-> (build/build-network mnist-description-with-toys)
                          (traverse/bind-input :input-1 :data)
                          (traverse/bind-output-train :softmax-1 :labels))
        gradient-descent (->> (traverse/network->gradient-descent built-network)
                              realize-traversals)
        inference-mem (->> (traverse/network->inference built-network)
                           realize-traversals)
        inference-none (->> (traverse/network->inference built-network
                                                         :optimise-type :none)
                            realize-traversals)]
    (is (= 434280 (get built-network :parameter-count)))
    (is (= 434280 (->> (get-in built-network [:layer-graph :buffers])
                       (map (comp m/ecount second))
                       (reduce +))))
    (is (= [nil nil]
           (minimal-diff
            [{:id :input-1, :incoming [{:id :input-1-input-0, :input-idx 0}], :outgoing :input-1}
             {:id :dropout-1, :incoming [:input-1], :outgoing :dropout-1}
             {:id :convolutional-1, :incoming [:dropout-1], :outgoing :convolutional-1}
             {:id :max-pooling-1, :incoming [:convolutional-1], :outgoing :max-pooling-1}
             {:id :relu-1, :incoming [:max-pooling-1], :outgoing :relu-1}
             {:id :dropout-2, :incoming [:relu-1], :outgoing :dropout-2}
             {:id :convolutional-2, :incoming [:dropout-2], :outgoing :convolutional-2}
             {:id :max-pooling-2, :incoming [:convolutional-2], :outgoing :max-pooling-2}
             {:id :relu-2, :incoming [:max-pooling-2], :outgoing :relu-2}
             {:id :dropout-3, :incoming [:relu-2], :outgoing :dropout-3}
             {:id :batch-normalization-1, :incoming [:dropout-3], :outgoing :batch-normalization-1}
             {:id :linear-1, :incoming [:batch-normalization-1], :outgoing :linear-1}
             {:id :relu-3, :incoming [:linear-1], :outgoing :relu-3}
             {:id :dropout-4, :incoming [:relu-3], :outgoing :dropout-4}
             {:id :linear-2, :incoming [:dropout-4], :outgoing :linear-2}
             {:id :softmax-1, :incoming [:linear-2], :outgoing :softmax-1}]
            (get gradient-descent :forward)))
        (is (= [nil nil ]
               (minimal-diff
                {:batch-normalization-1 {:size 800},
                 :convolutional-1 {:size 11520},
                 :convolutional-2 {:size 3200},
                 :dropout-1 {:size 784},
                 :dropout-2 {:size 2880},
                 :dropout-3 {:size 800},
                 :dropout-4 {:size 500},
                 :input-1 {:size 784},
                 :input-1-input-0 {:inputs {0 784}, :size 784},
                 :linear-1 {:size 500},
                 :linear-2 {:size 10},
                 :max-pooling-1 {:size 2880},
                 :max-pooling-2 {:size 800},
                 :relu-1 {:size 2880},
                 :relu-2 {:size 800},
                 :relu-3 {:size 500},
                 :softmax-1 {:output {0 10}, :size 10}}
                (get gradient-descent :buffers)))))

    (is (= [nil nil]
           (minimal-diff
            [{:id :input-1, :incoming [{:id :input-1-input-0, :input-idx 0}], :outgoing :input-1}
             {:id :convolutional-1, :incoming [:input-1], :outgoing :input-1-input-0}
             {:id :max-pooling-1, :incoming [:input-1-input-0], :outgoing :input-1}
             {:id :relu-1, :incoming [:input-1], :outgoing :input-1-input-0}
             {:id :convolutional-2, :incoming [:input-1-input-0], :outgoing :input-1}
             {:id :max-pooling-2, :incoming [:input-1], :outgoing :input-1-input-0}
             {:id :relu-2, :incoming [:input-1-input-0], :outgoing :input-1}
             {:id :batch-normalization-1, :incoming [:input-1], :outgoing :input-1-input-0}
             {:id :linear-1, :incoming [:input-1-input-0], :outgoing :input-1}
             {:id :relu-3, :incoming [:input-1], :outgoing :input-1-input-0}
             {:id :linear-2, :incoming [:input-1-input-0], :outgoing :input-1}
             {:id :softmax-1, :incoming [:input-1], :outgoing :input-1-input-0}]
            (get inference-mem :forward))))
    (is (= {:input-1 {:id :input-1, :size 3200}, :input-1-input-0 {:id :input-1-input-0, :inputs {0 784}, :size 11520}}
           (get inference-mem :buffers)))
    (is (= [nil nil]
           (minimal-diff
            [{:id :input-1, :incoming [{:id :input-1-input-0, :input-idx 0}], :outgoing :input-1}
             {:id :convolutional-1, :incoming [:input-1], :outgoing :convolutional-1}
             {:id :max-pooling-1, :incoming [:convolutional-1], :outgoing :max-pooling-1}
             {:id :relu-1, :incoming [:max-pooling-1], :outgoing :relu-1}
             {:id :convolutional-2, :incoming [:relu-1], :outgoing :convolutional-2}
             {:id :max-pooling-2, :incoming [:convolutional-2], :outgoing :max-pooling-2}
             {:id :relu-2, :incoming [:max-pooling-2], :outgoing :relu-2}
             {:id :batch-normalization-1, :incoming [:relu-2], :outgoing :batch-normalization-1}
             {:id :linear-1, :incoming [:batch-normalization-1], :outgoing :linear-1}
             {:id :relu-3, :incoming [:linear-1], :outgoing :relu-3}
             {:id :linear-2, :incoming [:relu-3], :outgoing :linear-2}
             {:id :softmax-1, :incoming [:linear-2], :outgoing :softmax-1}]
            (get inference-none :forward))))
    (is (= [nil nil]
           (minimal-diff
            {:batch-normalization-1 {:size 800},
             :convolutional-1 {:size 11520},
             :convolutional-2 {:size 3200},
             :input-1 {:size 784},
             :input-1-input-0 {:inputs {0 784}, :size 784},
             :linear-1 {:size 500},
             :linear-2 {:size 10},
             :max-pooling-1 {:size 2880},
             :max-pooling-2 {:size 800},
             :relu-1 {:size 2880},
             :relu-2 {:size 800},
             :relu-3 {:size 500},
             :softmax-1 {:output {0 10}, :size 10}}
            (get inference-none :buffers))))))
