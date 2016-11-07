(ns caffe.core
  (:require [think.hdf5.core :as hdf5]
            [cortex.nn.caffe :as cortex-caffe]
            [think.resource.core :as resource]
            [clojure.java.io :as io])
  (:import [java.io StringReader]))


(defn caffe-h5->model
  [fname]
  (resource/with-resource-context
    (let [file-node (hdf5/open-file fname)
          file-children (hdf5/child-map file-node)
          prototxt-reader (-> (:model_prototxt file-children)
                              (hdf5/->clj)
                              :data
                              first
                              (StringReader.))]
      (cortex-caffe/load-text-caffe-file prototxt-reader))))
