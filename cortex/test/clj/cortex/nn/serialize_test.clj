(ns cortex.nn.serialize-test
  (:require
   [clojure.test :refer [deftest is are]]
   [clojure.core.matrix :as m]
   [cortex.nn.network :as net]
   [cortex.nn.spiral-test :as st]
   [cortex.nn.serialization :as cs])
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
