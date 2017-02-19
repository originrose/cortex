(ns cortex.verify.utils
  (:require [clojure.core.matrix :as m]
            [think.resource.core :as resource]
            [clojure.test :refer [deftest]])
  (:import [java.math BigDecimal MathContext]))



(defn abs-diff
  [a b]
  (m/esum (m/abs (m/sub a b))))


(defn mean-absolute-error
  [a b]
  (/ (m/esum (m/abs (m/sub a b)))
     (m/ecount a)))


(def epsilon 1e-6)


(defn about-there?
  ([a b eps]
   (< (abs-diff a b) eps))
  ([a b]
   (about-there? a b epsilon)))


(defn round-to-sig-figs
  ^double [^double lhs ^long num-sig-figs]
  (-> (BigDecimal. lhs)
      (.round (MathContext. num-sig-figs))
      (.doubleValue)))


(defn sig-fig-equal?
  [^double lhs ^double rhs ^long num-sig-figs]
  (= (round-to-sig-figs lhs num-sig-figs)
     (round-to-sig-figs rhs num-sig-figs)))
