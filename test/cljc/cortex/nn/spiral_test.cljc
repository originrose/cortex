(ns cortex.nn.spiral-test
  (:require
    [clojure.core.matrix :as m]
    [cortex.optimise :as opt]
    [cortex.util :as util]
    [cortex.nn.network :as net]
    [cortex.nn.core :as core]
    [cortex.nn.layers :as layers]))

(def num-points 200)
(def num-classes 3)

(m/set-current-implementation #?(:clj :vectorz :cljs :thing-ndarray))

(defn create-spiral
  [start-theta end-theta start-radius end-radius num-items]
  (let [theta-increments (/ (- end-theta start-theta) num-items)
        radius-increments (/ (- end-radius start-radius) num-items)]
    (mapv (fn [idx]
            (let [theta (+ start-theta (* theta-increments idx))
                  radius (+ start-radius (* radius-increments idx))]
              (m/mul! (m/array [(Math/sin theta) (Math/cos theta)]) radius)))
          (range num-items))))


(defn create-spiral-from-index
  [idx]
  (let [start-theta (/ idx Math/PI)
        end-theta (+ start-theta 4)
        start-radius 0.1
        end-radius 2.0]
    (create-spiral start-theta end-theta start-radius end-radius num-points)))


(def all-data (into [] (mapcat create-spiral-from-index (range num-classes))))
(def all-labels (into [] (mapcat #(repeat num-points (m/array (assoc (into [] (repeat num-classes 0)) % 1))) (range num-classes))))
(def loss-fn (opt/mse-loss))
(def hidden-layer-size 10)


(defn softmax-network
  []
  (core/stack-module [(layers/linear-layer 2 hidden-layer-size)
                      (layers/relu [hidden-layer-size])
                      (layers/linear-layer hidden-layer-size hidden-layer-size)
                      (layers/relu [hidden-layer-size])
                      (layers/linear-layer hidden-layer-size num-classes)
                      (layers/softmax [num-classes])
                      ]))

(defn create-optimizer
  [network]
  (opt/adadelta-optimiser (core/parameter-count network)))


(defn train-and-evaluate
  []
  (let [network (softmax-network)
        optimizer (create-optimizer network)
        network (net/train network optimizer loss-fn all-data all-labels 10 100)
        score (net/evaluate-softmax network all-data all-labels)]
    [score network]))

(defonce score-network (atom nil))

(defn get-score-network
  []
  (when-not @score-network
    (reset! score-network (train-and-evaluate)))
  @score-network)
