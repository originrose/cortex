(ns cortex.suite.classification
  (:require [cortex.dataset :as ds]
            [clojure.java.io :as io]
            [think.parallel.core :as parallel])
  (:import [java.io File]))


(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)


(defn infinite-file-label-pairs
  "Given a directory with subdirs named after labels, produce a vector
of N infinite arrays each with random ordering of files to labels.  Then in order
to create balanced training classes using partition along with interleave."
  [dirname]
  (mapv #(mapcat identity
                 (repeatedly #(shuffle (map vector
                                            (repeat (.getName ^File %))
                                            (.listFiles ^File %)))))
        (.listFiles ^File (io/file dirname))))



(defn infinite-balanced-patches
  "Create a balanced infinite sequence of labels and patches.
Assumptions are that the src-item-seq is balanced by class
and that the transformation from src item to a list of [label, patch]
maintaines that class balance meaning applying that either produces
roughly uniform sequences of label->patch.  The transformation from
src item -> patch will be done in a threaded pool and fed into a queue
with the result further slighly interleaved.

The final results will be pulled out of the sequence and shuffled so assuming
your epoch element count is large enough the result will still be balanced and
random regardless of this function's specific behavior for any specific src item."
  [src-item-seq item->patch-seq-fn & {:keys [queue-size]
                                      :or {queue-size 1000}}]
  (let [{:keys [sequence shutdown-fn]} (parallel/queued-sequence queue-size
                                                                 (* 2 (.availableProcessors Runtime/getRuntime))
                                                                 item->patch-seq-fn
                                                                 src-item-seq)]
    :observations (mapcat identity sequence)
    :shutdown-fn shutdown-fn))
