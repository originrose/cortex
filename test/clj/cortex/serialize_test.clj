(ns cortex.serialize-test
  (:require
   [clojure.test :refer [deftest is are]]
   [cortex.optimise :as opt]
   [clojure.core.matrix :as m]
   [clojure.core.matrix.random :as randm]
   [cortex.util :as util]
   [cortex.network :as net]
   [cortex.core :as core]
   [cortex.layers :as layers]
   [cortex.protocols :as cp]
   [clojure.pprint]
   [cortex.spiral-test :as st]
   [cortex.serialization :as cs])
  (:import [java.io ByteArrayOutputStream ByteArrayInputStream]))


;;Test that we can save a nn to simpler datastructure
(deftest serialize-map-test
  (let [[score network] (st/get-score-network)
        map-network (cs/module->map network)
        new-network (cs/map->module map-network)
        new-score (net/evaluate-softmax new-network st/all-data st/all-labels)]
    (is (< (Math/abs (- new-score score)) 0.0001))))


(deftest serialize-to-stream-test
  (let [[score network] (st/get-score-network)
        output-stream (ByteArrayOutputStream.)
        _ (cs/write-network! network output-stream)
        byte-data (.toByteArray output-stream)
        input-stream (ByteArrayInputStream. byte-data)
        new-network (cs/read-network! input-stream)
        new-score (net/evaluate-softmax new-network st/all-data st/all-labels)]
    (is (< (Math/abs (- new-score score)) 0.0001))))
