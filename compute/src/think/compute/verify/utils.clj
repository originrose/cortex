(ns think.compute.verify.utils
  (:require [clojure.core.matrix :as m]
            [resource.core :as resource]
            [clojure.test :refer [deftest]])
  (:import [java.math BigDecimal MathContext]))


(defn test-wrapper
  [test-fn]
  (resource/with-resource-context
    (test-fn)))


(def ^:dynamic *datatype* :double)


(defmacro def-double-float-test
  [test-name & body]
  (let [double-test-name (str test-name "-d")
        float-test-name (str test-name "-f")]
   `(do
      (deftest ~(symbol double-test-name)
        (with-bindings {#'*datatype* :double}
          ~@body))
      (deftest ~(symbol float-test-name)
        (with-bindings {#'*datatype* :float}
          ~@body)))))


(defmacro def-all-dtype-test
  [test-name & body]
  (let [double-test-name (str test-name "-d")
        float-test-name (str test-name "-f")
        long-test-name (str test-name "-l")
        int-test-name (str test-name "-i")
        short-test-name (str test-name "-s")
        byte-test-name (str test-name "-b")
        ]
   `(do
      (deftest ~(symbol double-test-name)
        (with-bindings {#'*datatype* :double}
          ~@body))
      (deftest ~(symbol float-test-name)
        (with-bindings {#'*datatype* :float}
          ~@body))
      (deftest ~(symbol long-test-name)
        (with-bindings {#'*datatype* :long}
          ~@body))
      (deftest ~(symbol int-test-name)
        (with-bindings {#'*datatype* :int}
          ~@body))
      (deftest ~(symbol short-test-name)
        (with-bindings {#'*datatype* :short}
          ~@body))
      (deftest ~(symbol byte-test-name)
        (with-bindings {#'*datatype* :byte}
          ~@body)))))


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
  ([^double lhs ^double rhs ^long num-sig-figs]
   (= (round-to-sig-figs lhs num-sig-figs)
      (round-to-sig-figs rhs num-sig-figs)))
  ([^double lhs ^double rhs]
   (sig-fig-equal? lhs rhs (if (= *datatype* :double)
                             4
                             2))))
