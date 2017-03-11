(ns cortex.verify.tensor
  (:require [cortex.tensor :as ct]
            [cortex.compute.cpu.driver :as cpu-driver]
            [cortex.compute.driver :as drv]
            [clojure.test :refer :all]
            [clojure.core.matrix :as m]))


(defmacro tensor-context
  [driver datatype & body]
  `(with-bindings {#'ct/*stream* (drv/create-stream ~driver)
                   #'ct/*datatype* ~datatype}
     ~@body))


(defn assign-constant!
  [driver datatype]
  (tensor-context
   driver datatype
   (let [tensor (ct/->tensor (partition 3 (range 9)))]
     (is (= (ct/ecount tensor) 9))
     (is (m/equals (range 9)
                   (ct/to-double-array tensor)))
     (ct/assign! tensor 1)
     (is (m/equals (repeat 9 1)
                   (ct/to-double-array tensor)))

     (let [rows (ct/rows tensor)
           columns (ct/columns tensor)]
       (doseq [row rows]
         (ct/assign! row 2))
       (is (m/equals (repeat 9 2)
                     (ct/to-double-array tensor)))
       (doseq [col columns]
         (ct/assign! col 3))
       (is (m/equals (repeat 9 3)
                     (ct/to-double-array tensor)))))))
