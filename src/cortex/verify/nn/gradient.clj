(ns cortex.verify.nn.gradient
  (:require [clojure.test :refer :all]
            [clojure.core.matrix :as m]
            [cortex.optimize :as opt]
            [cortex.nn.execute :as execute]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
            [cortex.nn.traverse :as traverse]
            [cortex.verify.utils :as utils]
            [cortex.verify.nn.layers :as verify-layers]
            [cortex.verify.nn.data :as verify-data]
            [cortex.loss.core :as loss]
            [cortex.gaussian :as gaussian]
            [cortex.util :as cortex-util]
            [cortex.loss.yolo2 :as yolo2]))


(defn close-enough?
  [lhs-vec rhs-vec]
  (empty? (remove #(utils/sig-fig-equal? (first %) (second %))
                  (map vector lhs-vec rhs-vec))))


(defn check-gradients
  [gradient-vector]
  (doseq [{:keys [gradient numeric-gradient buffer-id]} gradient-vector]
    (let [mae (utils/mean-absolute-error (m/as-vector gradient) (m/as-vector numeric-gradient))]
      (is (< mae 1e-6)
          ;;Only build expensive message if we have to.
          (format "Gradient %s failed gradient check (MAE: %s) :\n%s." (name buffer-id) mae
                  (with-out-str
                    (clojure.pprint/pprint
                     (mapv vector
                           (m/eseq gradient)
                           (m/eseq numeric-gradient)))))))))


(defn add-id-to-desc-list
  [network]
  (let [network (-> network flatten vec)]
   (-> network
       (assoc-in [0 :id] :input)
       (assoc-in [(- (count network) 1) :id] :test))))


(defn generate-gradients
  [context description dataset epsilon batch-size labels-key]
  (execute/with-compute-context context
    (-> (network/linear-network description)
        (execute/generate-numeric-gradients context batch-size dataset epsilon)
        (verify-layers/unpack-network :test))))


(defn get-gradients
  [& args]
  (let [{:keys [input-gradient numeric-input-gradient parameters] :as gen-result}
        (apply generate-gradients args)]
    (concat [{:buffer-id :input
              :gradient input-gradient
              :numeric-gradient numeric-input-gradient}]
            (->> parameters
                 (map second)
                 (filter #(get-in % [:description :gradients?]))
                 (map (fn [{:keys [description parameter buffer] :as entry}]
                        (merge parameter buffer)))))))


(defn- data->dataset
  [input output batch-size & {:keys [input-name output-name]
                              :or {input-name :data
                                   output-name :test}}]
  (->> (map (fn [data label]
              {:data data
               :test label})
            input
            output)
       (take batch-size)))


(defn corn-gradient
  [context]
  (let [batch-size 2]
    (-> (get-gradients context
                       [(layers/input 1 1 2 :id :data)
                        (layers/linear 1 :id :test)]
                       (data->dataset verify-data/CORN-DATA verify-data/CORN-LABELS batch-size)
                       1e-4 batch-size :test)
        check-gradients)))


(defn batch-normalization-gradient
  [context]
  (let [batch-size 10
        input-size 20
        input (repeatedly batch-size
                          #(-> (repeatedly input-size gaussian/rand-gaussian)
                               double-array
                               (gaussian/ensure-gaussian! 5 20)))
        output input]
    (-> (get-gradients context
                       [(layers/input 1 1 input-size :id :data)
                        (layers/batch-normalization :ave-factor 1.0 :id :test)]
                       (data->dataset input output batch-size)
                       1e-4 batch-size :test)
        check-gradients)))


(defn lrn-gradient
  [context]
  (let [batch-size 2
        input-dim 2
        input-num-pixels (* input-dim input-dim)
        num-input-channels 3
        lrn-n 3
        n-input (* num-input-channels input-num-pixels)
        input (repeat batch-size (range n-input))
        output input]
    (-> (get-gradients context
                       [(layers/input input-dim input-dim num-input-channels :id :data)
                        (layers/local-response-normalization :k 1 :n lrn-n
                                                             :alpha 1.0 :beta 0.75
                                                             :id :test)]
                       (data->dataset input output batch-size)
                       1e-4 batch-size :test)
        check-gradients)))


(defn prelu-gradient
  [context]
  (let [batch-size 10
        input-dim 3
        n-channels 4
        input-num-pixels (* input-dim input-dim)
        n-input (* input-num-pixels n-channels)
        input (repeat batch-size (repeat (quot n-input 2) [-1 1]))
        output input]
    (-> (get-gradients context
                       [(layers/input input-dim input-dim n-channels :id :data)
                        (layers/prelu :id :test)]
                       (data->dataset input output batch-size)
                       1e-4 batch-size :test)
        check-gradients)))


(defn concat-gradient
  [context]
  (let [batch-size 4
        item-count 5
        num-inputs 2
        inputs (->> (range (* batch-size item-count num-inputs))
                    ;;Input-count length vectors
                    (partition item-count)
                    ;;Groups of 2
                    (partition num-inputs)
                    ;;destructure and make into maps.
                    (map (fn [[right-input left-input]]
                           {:right right-input
                            :left left-input})))

        outputs (repeat batch-size {:test (repeat (* num-inputs item-count) 1)})
        dataset (map merge inputs outputs)]
    (-> (get-gradients context
                       [(layers/input item-count 1 1 :id :right)
                        (layers/input item-count 1 1 :parents [] :id :left)
                        (layers/concatenate :parents [:left :right]
                                            :id :test)]
                       dataset
                       1e-4 batch-size :test)
        check-gradients)))


(defn split-gradient
  [context]
  (let [batch-size 4
        input-size 5
        num-outputs 2
        inputs (->> (range (* batch-size input-size))
                    (partition input-size)
                    (map (fn [input]
                           {:data input})))
        outputs (repeat batch-size
                        {:output-1 (repeat input-size 1)
                         :output-2 (repeat input-size 1)})
        dataset (map merge inputs outputs)]
    (-> (get-gradients context
                       [(layers/input input-size 1 1 :id :data)
                        (layers/split :id :test)
                        (layers/split :id :output-1)
                        (layers/split :parents [:test] :id :output-2)]
                       dataset
                       1e-4 batch-size :test)
        check-gradients)))


(defn- join-dataset
  [batch-size]
  (let [input-counts [3 4 5]
        num-inputs (count input-counts)
        inputs (->> (repeat [-1 -2 1 4])
                    flatten
                    (take (* batch-size (apply + input-counts)))
                    (cortex-util/super-partition (->> (repeat batch-size input-counts)
                                                      flatten))
                    (partition num-inputs)
                    (map (fn [[left middle third]]
                           {:left left
                            :middle middle
                            :third third})))
        output-count (apply max input-counts)
        outputs (->> (repeat (* batch-size output-count) 1)
                     (partition output-count)
                     (map #(hash-map :test %)))]
    (map merge inputs outputs)))


(defn join-+-gradient
  [context]
  (let [batch-size 4]
    (-> (get-gradients context
                       [(layers/input 3 1 1 :id :left)
                        (layers/input 4 1 1 :parents [] :id :middle)
                        (layers/input 5 1 1 :parents [] :id :right)
                        (layers/join :parents [:left :middle :right]
                                     :id :test)]
                       (join-dataset batch-size)
                       1e-4 batch-size :test)
        check-gradients)))


(defn join-*-gradient
  [context]
  (let [batch-size 4]
        (-> (get-gradients context
                           [(layers/input 3 1 1 :id :left)
                            (layers/input 4 1 1 :parents [] :id :middle)
                            (layers/input 5 1 1 :parents [] :id :right)
                            (layers/join :parents [:left :middle :right]
                                         :operation :*
                                         :id :test)]
                           (join-dataset batch-size)
                           1e-4 batch-size :test)
            check-gradients)))


(defn censor-gradient
  [context]
  (let [batch-size 1
        inputs [4.0 6.0]
        outputs [-0.5 Double/NaN]
        input-dim (count inputs)
        output-dim (count outputs)]
    (-> (get-gradients context
                       [(layers/input input-dim 1 1 :id :data)
                        (layers/linear output-dim
                                       :id :test
                                       :censor-loss {:labels {:type :stream
                                                              :stream :test}
                                                     :nan-zero-labels {:stream :test}
                                                     :gradient-masks {:stream :test}})]
                       [{:data inputs
                         :test outputs}]
                       1e-4
                       batch-size
                       :test)
        check-gradients)))


(defn yolo-gradient
  [context]
  (let [batch-size 2
        grid-x 3
        grid-y 3
        n-classes 4
        anchors (partition 2 [9.42 5.11 16.62 10.52])
        anchor-count (first (m/shape anchors))
        truth (-> (yolo2/realize-label {:filename "2012_004196.png", :width 500, :height 375,
                                        :segmented? false, :objects [{:class "person", :bounding-box [143 7 387 375]}]}
                                       {:grid-x 3 :grid-y 3 :anchor-count anchor-count :class-count 4
                                        :class-name->label (constantly [0 0 1 0])})
                  :label)
        input-size (* grid-x grid-y anchor-count (+ 5 n-classes))
        input (partition input-size (vec (repeatedly (* input-size batch-size) rand)))
        output (repeat batch-size (m/to-double-array truth))]
    (-> (get-gradients context
                       [(layers/input input-size 1 1 :id :data)
                        (layers/relu :id :test
                                     :yolo2 {:grid-x 3
                                             :grid-y 3
                                             :anchors anchors
                                             :labels {:type :stream
                                                      :stream :test}})]
                       (data->dataset input output batch-size)
                       1e-4 batch-size :test)
        (check-gradients))))
