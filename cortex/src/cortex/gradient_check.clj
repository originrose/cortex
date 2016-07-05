(ns cortex.gradient-check
  (:require [clojure.core.matrix.macros :refer [c-for]]
            [clojure.core.matrix :as m]))


(defn calc-numeric-gradient
  [param-vec test-fn & {:keys [epsilon]
                        :or {epsilon 1e-4}}]
  (let [num-elems (m/ecount param-vec)
        retval (double-array num-elems)]
    (c-for [idx 0 (< idx num-elems) (inc idx)]
           (let [orig-val (m/mget param-vec idx)
                 pos-vec (m/mset param-vec idx (+ orig-val epsilon))
                 neg-vec (m/mset param-vec idx (- orig-val epsilon))
                 pos-val (test-fn pos-vec)
                 neg-val (test-fn neg-vec)]
             (aset retval idx (/ (- pos-val neg-val) (* 2.0 epsilon)))))
    (vec retval)))
