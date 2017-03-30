(ns cortex.compute.verify.utils
  (:require [clojure.core.matrix :as m]
            [think.resource.core :as resource]
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


(defmacro def-cas-dtype-test
  "These test are only valid for datatypes that support gpu CAS operations."
  [test-name & body]
  (let [double-test-name (str test-name "-d")
        float-test-name (str test-name "-f")
        long-test-name (str test-name "-l")
        int-test-name (str test-name "-i")]
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
          ~@body)))))
