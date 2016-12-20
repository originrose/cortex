(ns cortex.verification.nn.layers
  "Verify that layers do actually produce their advertised results."
  (:require [cortex.nn.layers :as layers]
            [cortex.nn.build :as build]
            [cortex.nn.traverse :as traverse]
            [cortex.nn.execute :as execute]
            [clojure.test :refer :all]
            [clojure.core.matrix :as m]))


(defn forward-backward-test
  [context network test-layer-id batch-size input output-gradient]
  (let [network (execute/forward-backward context
                                          (assoc (build/build-network network)
                                                 :traversal
                                                 (traverse/network->gradient-descent network)
                                                 :batch-size
                                                 batch-size)
                                          input output-gradient)
        traverse (get network :traversal)
        test-node (get-in network [:layer-graph test-layer-id])
        parameter-descriptions (layers/get-parameter-descriptions test-node)
        parameters (mapv (fn [{:keys [key] :as description}]
                           (let [parameter (get test-node key)
                                 buffer (get-in network
                                                [:layer-graph :buffers
                                                 (get parameter :buffer-id)])]
                             {:description description
                              :parameter parameter
                              :buffer buffer}))
                         parameter-descriptions)
        {:keys [forward buffers]} traversal
        {:keys [incoming id outgoing]} (->> forward
                                            (filter #(= test-layer-id
                                                        (get % :id)))
                                            first)
        incoming-buffers (mapv buffers incoming)
        outgoing-buffers (mapv buffers outgoing)]
    {:network network
     :parameters parameters
     :incoming-buffers incoming-buffers
     :outgoing-buffers outgoing-buffers
     :output (get (first outgoing-buffers) :buffer)
     :input-gradient (get (first incoming-buffers :gradient))}))


(defn relu-activation
  [context]
  (let [item-count 10
        ;;sums to zero
        {:keys [output input-gradient]}
        (forward-backward-test context
                               [(layers/input item-count)
                                (layers/relu :id :test)]
                               :test
                               1
                               ;;input sums to zero
                               (flatten (repeat (/ item-count 2) [-1 1]))
                               ;;output-gradient sums to item-count
                               (repeat item-count 1))]
    (is (= (double (/ item-count 2))
           (m/esum output)))
    (is (= (double (/ item-count 2))
           (m/esum input-gradient)))))
