(ns caffe.core
  (:require [think.hdf5.core :as hdf5]
            [cortex.nn.description :as desc]
            [think.resource.core :as resource]
            [clojure.java.io :as io]
            [instaparse.core :as insta]
            [clojure.string :as string]
            [think.datatype.core :as dtype]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.macros :refer [c-for]]
            [think.compute.verify.import :as verify-import])
  (:import [java.io StringReader]))


(defn- parse-prototxt
  [parse-str]
  (let [retval
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
whitespace = #'\\s*'") parse-str)]
    retval))

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
               :caffe-top (keyword (:top layer))
               :caffe-layer layer)]
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
  (into {} (map (fn [[k v]]
                  [k (if (string? v)
                       (read-string v)
                       v)])
                item-map)))

(defmethod prototxt-layer->desc :Convolution
  [layer]
  (let [{:keys [kernel_size num_output pad stride group]
         :or {pad 0 stride 1 group 1} :as test-map}
        (-> (get layer :convolution_param)
            read-string-vals)
        assoc-group-fn #(assoc % :group group)]
    (->> (desc/convolutional kernel_size pad stride num_output)
         first
         assoc-group-fn
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
    (->> (desc/max-pooling kernel_size pad stride)
         first
         (add-layer-data-to-desc layer))))


(defmethod prototxt-layer->desc :LRN
  [layer]
  (let [{:keys [alpha beta local_size]
         :or {alpha 0.0001 beta 0.75 local_size 5}}
        (-> (get layer :lrn_param)
            read-string-vals)]
    (->> (desc/local-response-normalization
          :alpha alpha :beta beta :n local_size :k 1)
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

(defn- link-next-item
  [current-item bottom-map]
  (when-let [next-item (get bottom-map (:caffe-top current-item))]
    (cons next-item (lazy-seq (link-next-item next-item bottom-map)))))

(defn- ensure-doubles
  ^doubles [data]
  (if (not= :double (dtype/get-datatype data))
    (let [double-data (double-array (m/ecount data))]
      (dtype/copy! data 0 double-data 0 (m/ecount data))
      double-data)
    data))


(defn- to-core-matrix
  ([data ideal-shape ^long group]
   (let [^doubles data (ensure-doubles data)]
     ;;https://github.com/mikera/core.matrix/issues/299
     ;;The simple case of using m/reshape has serious performance issues.
     ;;In addition, we have to worry about 'group' which is a specific thing to
     ;;caffe networks:
     ;;http://caffe.berkeleyvision.org/tutorial/layers.html
     ;;My solution is to expand the weights such that they have zeros over the input they should not
     ;;care about.
     (case (count ideal-shape)
       1 data
       2 (let [[n-rows n-cols] ideal-shape
               group-rows (long (quot ^long n-rows group))
               ^"[[D" retval (make-array Double/TYPE n-rows (* n-cols group))]
           (c-for [row 0 (< row n-rows) (inc row)]
                  (let [group-copy-offset (* n-cols (quot row group-rows))]
                    (dtype/copy! data (* row n-cols) (aget retval row) group-copy-offset n-cols)))
           retval))))
  ([data ideal-shape]
   (to-core-matrix data ideal-shape 1)))


(defn- dim-shape->core-m-shape
  [weight-shape]
  (condp = (count weight-shape)
    1 weight-shape
    2 weight-shape
    4 [(first weight-shape) (reduce * (drop 1 weight-shape))]
    (throw (Exception. (format "Unexpected weight shape: %s" weight-shape)))))


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
                        (reduce recurse-parse-prototxt {}))
          layer-list (mapv prototxt-layer->desc (:layer prototxt))
          layer-map (into {} (map-indexed (fn [idx desc]
                                            [(:id desc) (assoc desc :layer-index idx)])
                                          layer-list))
          weight-children (hdf5/child-map (:model_weights file-children))
          layer-map (reduce
                     (fn [layer-map [layer-id weights-data]]
                       (if-let [target-desc (get layer-map layer-id)]
                         (let [weight-children (hdf5/child-map weights-data)
                               weight-id (keyword (str (name layer-id) "_W"))
                               bias-id (keyword (str (name layer-id) "_b"))
                               weights (hdf5/->clj (get weight-children weight-id))
                               bias (hdf5/->clj (get weight-children bias-id))]
                           (assoc layer-map layer-id
                                  (assoc target-desc
                                         :weights (to-core-matrix
                                                   (ensure-doubles (:data weights))
                                                   (dim-shape->core-m-shape
                                                    (:dimensions weights))
                                                   (long (or (:group target-desc) 1)))
                                         :bias (to-core-matrix
                                                (ensure-doubles (:data bias))
                                                (dim-shape->core-m-shape
                                                 (:dimensions bias))))))
                         (do
                           (println "Failed to find node for" layer-id)
                           layer-map)))
                     layer-map
                     weight-children)
          ;;caffe layers flow bottom to top (despite being listed top to bottom)
          ;;so a layer takes bottom and produces top.
          layer-output-map (group-by :caffe-top layer-list)
          layer-outputs (hdf5/child-map (:layer_outputs file-children))
          layer-id->output (reduce (fn [retval [layer-id node]]
                                     (if-let [output-group (get layer-output-map layer-id)]
                                       (assoc retval (:id (last output-group)) (ensure-doubles (:data (hdf5/->clj node))))
                                       (do
                                         (println "Failed to find layer for output:" layer-id)
                                         retval)))
                                   {}
                                   layer-outputs)
          model (vec (sort-by :layer-index (map second layer-map)))
          input-id (:id (first model))
          input (layer-id->output input-id)
          layer-outputs (mapv (fn [desc]
                                (layer-id->output (:id desc)))
                              (drop 1 model))
          model (mapv (fn [desc]
                        (if-let [output-vec (layer-id->output (:id desc))]
                          (assoc desc :caffe-output-size (count output-vec))
                          desc))
                      model)]
      {:prototxt prototxt
       :model model
       :input input
       :layer-outputs layer-outputs})))


(defn test-caffe-file
  [fname]
  (let [import-result (caffe-h5->model fname)]
    (println (format "Verifying %d layers" (count (:model import-result))))
    (verify-import/verify-model import-result)))
