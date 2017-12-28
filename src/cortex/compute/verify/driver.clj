(ns cortex.compute.verify.driver
  (:require [clojure.test :refer :all]
            [cortex.compute.driver :as drv]
            [cortex.compute.math :as math]
            [think.datatype.core :as dtype]
            [think.datatype.base :as dtype-base]
            [clojure.core.matrix :as m]))



(defn simple-stream
  [driver datatype]
  (drv/with-compute-device
    (drv/default-device driver)
    (let [stream (drv/create-stream)
          buf-a (drv/allocate-host-buffer driver 10 datatype)
          output-buf-a (drv/allocate-host-buffer driver 10 datatype)
          buf-b (drv/allocate-device-buffer 10 datatype)
          input-data (dtype/make-array-of-type datatype (range 10))
          output-data (dtype/make-array-of-type datatype 10)]
      (dtype/copy! input-data 0 buf-a 0 10)
      (dtype-base/set-value! buf-a 0 100.0)
      (dtype/copy! buf-a 0 output-data 0 10)
      (drv/copy-host->device stream buf-a 0 buf-b 0 10)
      (drv/memset stream buf-b 5 20.0 5)
      (drv/copy-device->host stream buf-b 0 output-buf-a 0 10)
      (drv/sync-stream stream)
      (dtype/copy! output-buf-a 0 output-data 0 10)
      (is (= [100.0 1.0 2.0 3.0 4.0 20.0 20.0 20.0 20.0 20.0]
             (mapv double output-data))))))

(defn make-buffer-fn
  [stream datatype]
   (fn [elem-seq]
     (drv/host-array->device-buffer stream
                                    (dtype/make-array-of-type datatype elem-seq))))

(defmacro backend-test-pre
  [driver datatype & body]
  `(drv/with-compute-device
     (drv/default-device ~driver)
     (let [~'stream (drv/create-stream)
           ~'make-buffer (make-buffer-fn ~'stream ~datatype)
           ~'->array (fn [buffer#] (drv/device-buffer->host-array ~'stream buffer#))]
       ~@body)))

(defn sum
  [driver datatype]
  (backend-test-pre driver datatype
   (let [values (flatten (repeat 4 (range 10)))
         buf-a (make-buffer values)
         buf-b (make-buffer values)]
     (math/sum stream 1.0 buf-a 2.0 buf-b)
     (is (= (vec (map #(double (* 3 %)) values))
            (vec (map double (->array buf-b))))))
   (let [values (flatten (repeat 4 (range 10)))
         buf-a (make-buffer values)
         buf-b (make-buffer (repeat 10 0))]
     (math/sum stream 1.0 buf-a 1.0 buf-b)
     (is (= (vec (map #(double (* 4 %)) (range 10)))
            (vec (map double (->array buf-b))))))))

(defn subtract
  [driver datatype]
  (backend-test-pre driver datatype
   (let [values (flatten (repeat 4 (range 10)))
         buf-a (make-buffer values)
         buf-b (make-buffer values)
         result (make-buffer (repeat (count values) 0))
         answer (m/sub values (m/mul values 2))]
     (math/subtract stream 1.0 buf-a 2.0 buf-b result)
     (is (= (vec (map double answer))
            (vec (map double (->array result))))))))

(defn gemv
  [driver datatype]
  (backend-test-pre driver datatype
   (let [values (vec (range 10))
         buf-a (make-buffer (flatten (repeat 4 values)))
         buf-b (make-buffer (flatten values))
         buf-c (make-buffer (repeat 4 0))
         a-row-count 4
         a-col-count 10
         answer (m/dot values values)]
     (math/gemv stream
      false a-row-count a-col-count
      1.0 buf-a a-col-count
      buf-b 1
      0.0 buf-c 1)
     (is (= (vec (map double (flatten (repeat 4 answer))))
            (vec (map double (->array buf-c))))))))

(defn mul-rows
  [driver datatype]
  (backend-test-pre driver datatype
   (let [values (vec (range 10))
         buf-a (make-buffer (flatten (repeat 4 values)))
         buf-b (make-buffer (range 4))
         buf-c (make-buffer (repeat 40 0))
         answer (concat (m/mul values 0)
                        (m/mul values 1)
                        (m/mul values 2)
                        (m/mul values 3))]
     (math/mul-rows stream 4 10 buf-a 10 buf-b 1 buf-c 10)
     (is (= (vec (map double answer))
            (vec (map double (->array buf-c))))))))

(defn elem-mul
  [driver datatype]
  (backend-test-pre driver datatype
   (let [values (vec (range 10))
         buf-a (make-buffer values)
         buf-b (make-buffer values)
         answer (mapv #(* % % 2.0) values)]
     (math/elem-mul stream 2.0 buf-a 1 buf-b 1 buf-b 1)
     (is (= (vec (map double answer))
            (vec (map double (->array buf-b))))))))


(defn close-enough
  [veca vecb]
  (< (m/distance veca vecb)
     0.001))

(defn l2-constraint-scale
  [driver datatype]
  (backend-test-pre driver datatype
   (let [values (mapv #(* % %) (range 10))
         buf-a (make-buffer values)
         answer [1 1 1 1 1 1 (/ 5 6) (/ 5 7) (/ 5 8) (/ 5 9)]]
     (math/l2-constraint-scale stream buf-a 1 5)
     (is (close-enough (mapv double answer)
                       (mapv double (->array buf-a)))))))
