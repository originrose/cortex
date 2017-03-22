(ns caffe.core
  (:require [think.hdf5.core :as hdf5]
            [think.resource.core :as resource]
            [clojure.java.io :as io]
            [instaparse.core :as insta]
            [clojure.string :as string]
            [think.datatype.core :as dtype]
            [clojure.core.matrix :as m]
            [clojure.core.matrix.macros :refer [c-for]]
            [cortex.verify.nn.import :as verify-import]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
            [cortex.nn.execute :as execute]
            [taoensso.nippy :as nippy]
            [cortex.nn.traverse :as traverse])
  (:import [java.io StringReader FileOutputStream]))


(defn- trim-till-end
  [^String line]
  (let [octothorpe-index (.indexOf line "#")]
    (if-not (= octothorpe-index -1)
      (.substring line 0 octothorpe-index)
      line)))


(defn- strip-comments
  [parse-str]
  (let [retval
        (->> parse-str
             string/split-lines
             (map trim-till-end)
             (string/join "\n"))]
    retval))


(defn- parse-prototxt
  [parse-str]
  (let [parser (insta/parser
           "<model> = parameter (parameter)* <whitespace>
<parameter> = <whitespace> (param-with-value | complex-parameter)
param-with-value = word <(':')?> <whitespace> value
begin-bracket = '{'
end-bracket = '}'
complex-parameter = <whitespace> word <(':')?> <whitespace> <begin-bracket>
<whitespace> model <whitespace> <end-bracket>
<value> = (word | <quote> word <quote>)
quote = '\"'
<word> = #'[\\w\\.-]+'
whitespace = #'\\s*'")
        retval (-> parse-str
                   strip-comments
                   parser)]
    (when (map? retval)
      (throw (ex-info "parse-error"
                      retval)))
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
       1 (layers/input (first layer-shape))
       4 (layers/input (nth layer-shape 3) (nth layer-shape 2) (nth layer-shape 1))
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
    (->> (layers/convolutional kernel_size pad stride num_output)
         first
         assoc-group-fn
         (add-layer-data-to-desc layer))))


(defmethod prototxt-layer->desc :ReLU
  [layer]
  (->> (layers/relu)
       first
       (add-layer-data-to-desc layer)))


(defmethod prototxt-layer->desc :Pooling
  [layer]
  (let [{:keys [kernel_size pad stride]
         :or {pad 0 stride 1}}
        (-> (get layer :pooling_param)
            read-string-vals)]
    (->> (layers/max-pooling kernel_size pad stride)
         first
         (add-layer-data-to-desc layer))))


(defmethod prototxt-layer->desc :LRN
  [layer]
  (let [{:keys [alpha beta local_size]
         :or {alpha 0.0001 beta 0.75 local_size 5}}
        (-> (get layer :lrn_param)
            read-string-vals)]
    (->> (layers/local-response-normalization
          :alpha alpha :beta beta :n local_size :k 1)
         first
         (add-layer-data-to-desc layer))))


(defmethod prototxt-layer->desc :InnerProduct
  [layer]
  (let [num_output (read-string (get-in layer [:inner_product_param :num_output]))]
    (->> (layers/linear num_output)
         first
         (add-layer-data-to-desc layer))))


(defmethod prototxt-layer->desc :Dropout
  [layer]
  (let [dropout_ratio (read-string (get-in layer [:dropout_param :dropout_ratio]))]
    (->> (layers/dropout dropout_ratio)
         first
         (add-layer-data-to-desc layer))))


(defmethod prototxt-layer->desc :Softmax
  [layer]
  (->> (layers/softmax)
       first
       (add-layer-data-to-desc layer)))


(defmethod prototxt-layer->desc :PReLU
  [layer]
  (->> (layers/prelu)
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
     ;;My solution is to expand the weights such that they have zeros over the
     ;;input they should not
     ;;care about.
     (case (count ideal-shape)
       1 data
       2 (let [[n-rows n-cols] ideal-shape
               group-rows (long (quot ^long n-rows group))
               ^"[[D" retval (make-array Double/TYPE n-rows (* n-cols group))]
           (c-for [row 0 (< row n-rows) (inc row)]
                  (let [group-copy-offset (* n-cols (quot row group-rows))]
                    (dtype/copy! data (* row n-cols) (aget retval row)
                                 group-copy-offset n-cols)))
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


(defn- check-for-input-params
  [prototxt]
  (when (and (contains? prototxt :input)
             (contains? prototxt :input_dim))
    (let [dimensions (get prototxt :input_dim)
          input-name (get prototxt :input)
          [batch channels height width] (mapv read-string dimensions)]
      (layers/input width height channels
                    :id (keyword input-name)
                    :caffe-bottom (keyword input-name)
                    :caffe-top (keyword input-name)))))


(defn caffe-h5->model
  [fname & {:keys [trim]}]
  (resource/with-resource-context
    (let [file-node (hdf5/open-file fname)
          file-children (hdf5/child-map file-node)
         prototxt (->> (:model_prototxt file-children)
                        (hdf5/->clj)
                        :data
                        first
                        parse-prototxt
                        (reduce recurse-parse-prototxt {}))
          layer-list (vec (concat (check-for-input-params prototxt)
                                  (map prototxt-layer->desc (:layer prototxt))))
          layer-map (->> (map-indexed (fn [idx desc]
                                        [(:id desc) (assoc desc :layer-index idx)])
                                      layer-list)
                         (into {})
                         (#(apply dissoc % trim)))
          weight-children (hdf5/child-map (:model_weights file-children))
          layer-map (reduce
                     (fn [layer-map [layer-id weights-data]]
                       (if-let [target-desc (get layer-map layer-id)]
                         (let [weight-children (hdf5/child-map weights-data)
                               weight-id (keyword (str (name layer-id) "_W"))
                               bias-id (keyword (str (name layer-id) "_b"))
                               weights (hdf5/->clj (get weight-children weight-id))
                               bias (when-let [bias-child (get weight-children bias-id)]
                                      (hdf5/->clj bias-child))
                               weight-key (if (= (get target-desc :type) :prelu)
                                            :neg-scale
                                            :weights)]
                           (assoc layer-map layer-id
                                  (cond-> (assoc target-desc
                                                 weight-key (to-core-matrix
                                                             (ensure-doubles (:data weights))
                                                             (dim-shape->core-m-shape
                                                              (:dimensions weights))
                                                             (long (or (:group target-desc) 1))))
                                    bias
                                    (assoc
                                           :bias
                                           (to-core-matrix
                                            (ensure-doubles (:data bias))
                                            (dim-shape->core-m-shape
                                             (:dimensions bias)))))))
                         (do
                           (println "Failed to find node for" layer-id)
                           layer-map)))
                     layer-map
                     weight-children)
          ;;caffe layers flow bottom to top (despite being listed top to bottom)
          ;;so a layer takes bottom and produces top.
          layer-output-map (group-by #(or (get % :caffe-top)
                                          (get % :caffe-bottom)) layer-list)
          layer-outputs (hdf5/child-map (:layer_outputs file-children))
          layer-id->output (reduce (fn [retval [layer-id node]]
                                     (if-let [output-group (get layer-output-map layer-id)]
                                       (assoc retval
                                              (:id (last output-group))
                                              (ensure-doubles (:data (hdf5/->clj node))))
                                       (do
                                         (println "Failed to find layer for output:" layer-id)
                                         retval)))
                                   {}
                                   layer-outputs)
          model (vec (sort-by :layer-index (map second layer-map)))
          model (mapv (fn [desc]
                        (if-let [output-vec (layer-id->output (:id desc))]
                          (assoc desc :caffe-output-size (count output-vec))
                          desc))
                      model)
          network (network/linear-network model)]
      (when-let [failures (seq (get network :verification-failures))]
        (let [ordered-nodes (->> (traverse/training-traversal network)
                                 :forward
                                 (mapv (fn [{:keys [incoming id outgoing]}]
                                           (get-in network [:compute-graph :nodes id]))))]
         (throw (ex-info "Verification failures detected:"
                         {:verification-failures (vec failures)
                          :layers ordered-nodes}))))
      {:prototxt prototxt
       :model network
       :layer-id->output layer-id->output})))


(defn test-caffe-file
  [fname & args]
  (let [import-result (apply caffe-h5->model fname args)]
    (println (format "Verifying %d layers" (count (get-in import-result [:model :compute-graph
                                                                         :nodes]))))
    (verify-import/verify-model (execute/compute-context) import-result)))


(defn import-and-write
  [^String fname & args]
  (let [import-result (apply caffe-h5->model fname args)]
    (if-let [failures (seq (verify-import/verify-model (execute/compute-context)
                                                       import-result))]
      (throw (ex-info "import verification failures: " (vec failures)))
      (let [f-stem (.substring fname 0 (.lastIndexOf fname "."))
            output-file (str f-stem ".nippy")
            nippy-data (nippy/freeze (get import-result :model))]
        (with-open [out-stream (FileOutputStream. output-file)]
          (.write out-stream nippy-data))))))
