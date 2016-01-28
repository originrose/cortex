(ns cortex.performance-test
  (:require [clojure.core.matrix :as m]
            [cortex.impl.layers :as impl]
            [cortex.core :as core]
            [clojure.test :refer [deftest is are]]
            [clojure.pprint]))


;;The point of this file is to highlight a couple operations in detail that
;;are currently causing perf issues and to investigate solutions to those issues.

(def implementations [:vectorz :clatrix])



(def image-dims [20 20 40 80 160 320])

(defn create-large-convolution-matrix-and-weights
  [implementation image-dim]
  (let [conv-config (impl/create-conv-layer-config
                     image-dim image-dim 5 5 0 0 1 1 1)
        kernel-count 20
        kernel-size (* 5 5)
        left-hand-side (m/array implementation (first (impl/get-gradient-convolution-sequence
                                                       conv-config)))
        right-hand-side (m/array implementation (map #(repeat kernel-size %) (range 1 (+ kernel-count 1))))]
    [left-hand-side right-hand-side]))

(def iter-count 200)

(defn time-matrix-multiply
  [implementation image-dim]
  (let [[left-hand-side right-hand-side] (create-large-convolution-matrix-and-weights implementation image-dim)]
    (print implementation)
    (time
     (dotimes [iter iter-count]
       (m/mmul left-hand-side (m/transpose right-hand-side))))))

(defn clatrix-in-place-mul
  [image-dim]
  (let [[left-hand-side right-hand-side] (create-large-convolution-matrix-and-weights :clatrix image-dim)
        left-shape (m/shape left-hand-side)
        right-shape (m/shape right-hand-side)
        ;;because the transpose, the right shape is the
        result (m/array :clatrix [(second left-shape) (second right-shape)])]
    (print "in-place clatrix")
    (time
     (dotimes [iter iter-count]
       (let [left-m (.me left-hand-side)
             right-m (.me (m/transpose right-hand-side))
             result-m (.me result)]
         (.mmuli left-m right-m result-m))))))


(deftest matrix-multiply-test
  (doseq [image-dim image-dims]
    (println "###" image-dim)
    (doseq [impl implementations]
      (time-matrix-multiply impl image-dim))
    (clatrix-in-place-mul image-dim)
    (println "###")))
