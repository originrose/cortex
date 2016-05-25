(ns cortex-gpu.util
  (:require [cortex-gpu.nn.cudnn :as cudnn]))


(defn get-or-allocate
  ([item key n-elems items-per-batch]
   (or (key item)
       (cudnn/new-array [n-elems] items-per-batch)))
  ([item key n-elems]
   (get-or-allocate item key n-elems 1)))


(defn assign-many->packed
  [many packed]
  (reduce (fn [offset item]
            (let [n-elems (cudnn/ecount item)]
              (cudnn/assign-async! item 0
                                   packed offset
                                   n-elems)
              (+ offset n-elems)))
          0
          many)
  packed)

(defn assign-packed->many
  [packed many]
  (reduce (fn [offset item]
            (let [n-elems (cudnn/ecount item)]
              (cudnn/assign-async! packed offset
                                   item 0
                                   n-elems)
              (+ offset n-elems)))
          0
          many))

(defn many-ecount
  [many]
  (let [total-ecount (reduce #(+ %1 (cudnn/ecount %2))
                             0
                             many)]
    total-ecount))


(defn zero-many
  [many]
  (doseq [item many]
    (cudnn/zero! item)))
