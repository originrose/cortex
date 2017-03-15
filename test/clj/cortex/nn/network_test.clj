(ns cortex.nn.network-test
  (:require
    [clojure.test :refer :all]
    [clojure.core.matrix :as m]
    [cortex.graph :as graph]
    [cortex.verify.nn.train :refer [CORN-DATA CORN-LABELS]]
    [cortex.nn.layers :as layers]
    [cortex.nn.execute :as execute]
    [cortex.nn.network :as network]))

(defn- corn-dataset
  []
  (mapv (fn [d l] {:data d :labels l})
        CORN-DATA CORN-LABELS))


(deftest specify-weights-bias
  (let [weight-data [[1 2][3 4]]
        bias-data [0 10]
        built-network (network/linear-network [(layers/input 2)
                                               (layers/linear
                                                2
                                                :weights weight-data
                                                :bias bias-data)])]
    (is (= (vec (m/eseq weight-data))
           (vec (m/eseq (get-in built-network [:compute-graph :buffers
                                               (get-in built-network
                                                       [:compute-graph :nodes
                                                        :linear-1 :weights :buffer-id])
                                               :buffer])))))
    (is (= (vec (m/eseq bias-data))
           (vec (m/eseq (get-in built-network [:compute-graph :buffers
                                               (get-in built-network
                                                       [:compute-graph :nodes
                                                        :linear-1 :bias :buffer-id])
                                               :buffer])))))))


(deftest generate-weights-bias
  (let [bias-data [0 0]
        built-network (network/linear-network [(layers/input 2)
                                              (layers/linear
                                               2)
                                              (layers/relu)])]
    (is (not (nil? (m/eseq (get-in built-network
                                   [:compute-graph :buffers
                                    (get-in built-network
                                            [:compute-graph :nodes
                                             :linear-1 :weights :buffer-id])
                                    :buffer])))))
    (is (m/equals (vec (m/eseq bias-data))
                  (vec (m/eseq (get-in built-network
                                       [:compute-graph :buffers
                                        (get-in built-network
                                                [:compute-graph :nodes
                                                 :linear-1 :bias :buffer-id])
                                        :buffer])))))))



(deftest prelu-initialization
  (let [built-network (network/linear-network [(layers/input 25 25 10)
                                              (layers/prelu)])]
    (is (= (vec (repeat 10 0.25))
           (vec (m/eseq (get-in built-network [:compute-graph :buffers
                                               (get-in built-network
                                                       [:compute-graph :nodes
                                                        :prelu-1 :neg-scale :buffer-id])
                                               :buffer])))))))


(deftest build-concatenate
  (let [network (network/linear-network [(layers/input 25 25 10 :id :right)
                                        (layers/input 500 1 1 :parents [] :id :left)
                                        (layers/concatenate :parents [:left :right] :id :concat)
                                        (layers/linear 10)])
        graph (network/network->graph network)
        concat-node (graph/get-node graph :concat)]
    (is (= (+ (* 25 25 10) 500)
           (graph/node->output-size concat-node)))
    (is (= (set [(assoc (first (graph/node->output-dimensions (graph/get-node graph :right)))
                        :id :right)
                 (assoc (first (graph/node->output-dimensions (graph/get-node graph :left)))
                        :id :left)])
           (set (graph/node->input-dimensions concat-node))))))


(defn test-run
  []
  (let [dataset (corn-dataset)]
    (execute/run [(layers/input 2 1 1 :id :data)
                  (layers/linear 1 :id :yield)]
                 dataset
                 :batch-size 1)))

(deftest classify-corn
  (testing "Ensure that we can run a simple classifier."
    (let [big-dataset (apply concat (repeatedly 5000 (fn []
                                                       (mapv (fn [{:keys [data labels]}]
                                                               {:labels (if (> (first labels) 50) 1 0)
                                                                :data data}) (corn-dataset))) ))]
      (loop [network (network/linear-network
                       [(layers/input 2 1 1 :id :data)
                        (layers/linear 2)
                        (layers/softmax :output-channels 2 :id :labels)])
             epoch 0]
        (if (> 20 epoch)
          (recur (cortex.nn.execute/train network big-dataset :batch-size 50) (inc epoch))
          network)))))
