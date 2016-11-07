(ns caffe.core
  (:require [think.hdf5.core :as hdf5]))

(defn hdf5-child-map
  [node]
  (into {} (map (fn [node-child]
                  [(keyword (hdf5/get-name node-child))
                   node-child])
                (hdf5/get-children node))))
