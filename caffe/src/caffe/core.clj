(ns caffe.core
  (:require [think.hdf5.core :as hdf5]
            [cortex.nn.description :as desc]
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

(defn- ->vec
  [data]
  (if (vector? data)
    data
    [data]))

(defmulti prototxt-layer->desc (fn [layer]
                                 (keyword (:type layer))))

(defn- add-layer-data-to-desc
  [layer desc]
  (let [retval
        (assoc desc
               :id (keyword (:name layer))
               :caffe-top (keyword (:top layer)))]
    (if (contains? layer :bottom)
      (assoc retval :caffe-bottom (keyword (:bottom layer)))
      retval)))

(defmethod prototxt-layer->desc :Input
  [layer]
  (let [layer-shape (mapv read-string (->vec
                                       (get-in layer [:input_param :shape :dim])))]

    (->>
     (condp = (count layer-shape)
       1 (desc/input (first layer-shape))
       4 (desc/input (nth layer-shape 3) (nth layer-shape 2) (nth layer-shape 1))
       (throw (Exception. (format "Unexpected layer shape %s %s"
                                  layer-shape layer))))
     first
     (add-layer-data-to-desc layer))))

(defn read-string-vals
  [item-map]
  (into {} (map (fn [[k v]] [k (read-string v)])
                item-map)))

(defmethod prototxt-layer->desc :Convolution
  [layer]
  (let [{:keys [kernel_size num_output pad stride group]
         :or [pad 0 stride 1 group 1]}
        (-> (get layer :convolution_param)
            read-string-vals)]
    (->> (desc/convolutional kernel_size 0 stride num_output :group group)
         first
         (add-layer-data-to-desc layer))))

(defmethod prototxt-layer->desc :ReLU
  [layer]
  (->> (desc/relu)
       first
       (add-layer-data-to-desc layer)))

(defmethod prototxt-layer->desc :Pooling
  [layer]
  (let [{:keys [kernel_size pad stride]
         :or {pad 0 stride 1}}
        (-> (get layer :pooling_param)
            read-string-vals)]
    (->> (desc/max-pooling kernel_size 0 stride)
         first
         (add-layer-data-to-desc layer))))


(defmethod prototxt-layer->desc :LRN
  [layer]
  (let [{:keys [alpha beta local_size k]
         :or {alpha 0.0001 beta 0.75 local_size 5 k 2}}
        (-> (get layer :lrn_param)
            read-string-vals)]
    (->> (desc/local-response-normalization
          :alpha alpha :beta beta :n local_size :k k)
         first
         (add-layer-data-to-desc layer))))


(defmethod prototxt-layer->desc :InnerProduct
  [layer]
  (let [num_output (read-string (get-in layer [:inner_product_param :num_output]))]
    (->> (desc/linear num_output)
         first
         (add-layer-data-to-desc layer))))


(defmethod prototxt-layer->desc :Dropout
  [layer]
  (let [dropout_ratio (read-string (get-in layer [:dropout_param :dropout_ratio]))]
    (->> (desc/dropout dropout_ratio)
         first
         (add-layer-data-to-desc layer))))


(defmethod prototxt-layer->desc :Softmax
  [layer]
  (->> (desc/softmax)
       first
       (add-layer-data-to-desc layer)))


(defmethod prototxt-layer->desc :default
  [layer]
  {:type :error
   :layer layer})


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
      (mapv prototxt-layer->desc (:layer prototxt)))))
