(ns caffe.core
  (:require [think.hdf5.core :as hdf5]
            [cortex.nn.caffe :as cortex-caffe]
            [think.resource.core :as resource]
            [clojure.java.io :as io]
            [instaparse.core :as insta]
            [clojure.string :as string])
  (:import [java.io StringReader]))

(defn- parse-prototxt
  [parse-str]
  ((insta/parser
    "<model> = parameter (parameter)* <whitespace>
<parameter> = <whitespace> (param-with-value | complex-parameter)
param-with-value = word <(':')?> <whitespace> value
begin-bracket = '{'
end-bracket = '}'
complex-parameter = <whitespace> word <(':')?> <whitespace> <begin-bracket>
<whitespace> model <whitespace> <end-bracket>
<value> = (word | <quote> word <quote>)
quote = '\"'
<word> = #'[\\w\\.]+'
whitespace = #'\\s*'") parse-str))

(defn- add-value-to-map
  [retval k v]
  (if (contains? retval k)
    (let [existing (get retval k)
          existing (if-not (vector? existing)
                     [existing]
                     existing)]
      (assoc retval k (conj existing v)))
    (assoc retval k v)))

(defn- recurse-parse-prototxt
  [retval value]
  (let [param-name (second value)]
    (if (= (first value) :complex-parameter)

      (add-value-to-map
       retval (keyword param-name)
       (reduce recurse-parse-prototxt
               {}
               (drop 2 value)))
      (add-value-to-map retval (keyword param-name) (nth value 2)))))


(defn caffe-h5->model
  [fname]
  (resource/with-resource-context
    (let [file-node (hdf5/open-file fname)
          file-children (hdf5/child-map file-node)
          prototxt (->> (:model_prototxt file-children)
                        (hdf5/->clj)
                        :data
                        first
                        parse-prototxt
                        (reduce recurse-parse-prototxt {}))]
      prototxt)))
