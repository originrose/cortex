(ns think.cortex.keras.core
  (:require [think.hdf5.core :as hdf5]
            [cortex.nn.description :as desc]
            [think.resource.core :as resource]
            [cheshire.core :as json]
            [clojure.java.io :as io]))


(defn read-model
  [fname]
  (json/parse-string (slurp fname) keyword))

(defmulti model-item->desc (fn [item]
                             (keyword (:class_name item))))

(defmethod model-item->desc :Convolution2D
  [{:keys [config]}]
  (let [[stride-x stride-y] (get config :subsample [1 1])
        [pad-x pad-y] (get config :padding [0 0])
        kernel-x (long (get config :nb_col))
        kernel-y (long (get config :nb_row))
        kernel-count (long (get config :nb_filter))
        id (keyword (get config :name))
        activation (keyword (get config :activation))
        conv-desc (first (desc/convolutional-expanded kernel-x kernel-y pad-x pad-y
                                                      stride-x stride-y kernel-count))
        conv-desc (assoc conv-desc :id id)]
    (if activation
      [conv-desc {:type activation}]
      [conv-desc])))


(defmethod model-item->desc :MaxPooling2D
  [{:keys [config]}]
  (let [[kernel-x kernel-y] (get config :pool_size)
        [stride-x stride-y] (get config :strides)]
    (desc/max-pooling kernel-x kernel-y 0 0 stride-x stride-y)))


(defn model->description
  [model]
  (when-not (= (:class_name model) "Sequential")
    (throw (Exception. "Only sequential models supported")))
  (let [[_ width height n-channels] (get-in model [:config 0 :config :batch_input_shape])
        ;;move zeropadding into convolution modules
        model-vector (reduce (fn [model-vector {:keys [class_name config] :as current}]
                               (if (and (= (keyword class_name) :Convolution2D)
                                        (= (keyword (get (last model-vector) :class_name))
                                           :ZeroPadding2D))
                                 (conj (vec (drop-last model-vector))
                                       (update-in current [:config]
                                                  #(merge (get (last model-vector)
                                                               :config)
                                                          %)))
                                 (conj model-vector current)))
                             [] (get-in model [:config]))]
    ;;TODO models with a single channel input and figure out planar vs. interleaved
    (vec
     (flatten (concat (desc/input width height n-channels)
                      (mapv model-item->desc model-vector))))))
