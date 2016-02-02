(ns cortex.generators
  "Namespace for test.check generator functions"
  (:require [clojure.test.check.generators :as gen]
            [clojure.test.check.generators :as tc]
            [clojure.core.matrix :as m]))

;; ===========================================
;; Implementation generators

(def gen-impl
  (gen/elements [:ndarray :persistent-vector :vectorz :object-array :double-array]))

;; ===========================================
;; Dimension, shape and scaling generators

(def gen-dims
  "A generator for dimensionalities (including zero). Grows quite slowly."
  (gen/scale #(Math/pow (double %) 0.333) gen/pos-int))

(def gen-s-pos-dims
  "A generator for dimensionalities greated than or equal to one. Grows quite slowly."
  (gen/such-that pos? gen-dims 100))

(defn gen-shape 
  "Creates a generator that returns valid core.matrix shapes for arrays, with strictly positive 
   dimension sizes. Grows roughly linearly in the number of elements."
  ([]
    (gen-shape gen-dims))
  ([gen-dims]
    (gen/bind gen-dims
              (fn [dims]
                (gen/vector (gen/scale #(Math/pow (double %) (/ 1.0 (double dims))) gen/s-pos-int) 
                            dims)))))

(def gen-matrix-shape
  (gen-shape (gen/return 2)))

(def gen-vector-shape
  (gen/fmap vector gen/s-pos-int))

;; ===========================================
;; Array generators

(defn gen-array
  "Creates a generator that returns arrays"
  ([g-shape g-element]
    (gen-array g-shape g-element :vectorz))
  ([g-shape g-element impl-or-g-impl]
    (let [g-impl (if (gen/generator? impl-or-g-impl) impl-or-g-impl (gen/return impl-or-g-impl))]
      (gen/bind 
        g-shape
        (fn [shape]
          (gen/bind 
            g-impl
            (fn [impl]
              (gen/fmap
                (fn [elts]
                  (m/array impl (m/reshape elts shape)))
                (gen/vector g-element (reduce * 1 shape))))))))))