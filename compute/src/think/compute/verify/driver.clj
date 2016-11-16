(ns think.compute.verify.driver
  (:require [clojure.test :refer :all]
            [think.compute.driver :as drv]
            [think.compute.math :as math]
            [think.datatype.core :as dtype]
            [clojure.core.matrix :as m]))



(defn simple-stream
  [device datatype]
  (let [stream (drv/create-stream device)
        buf-a (drv/allocate-host-buffer device 10 datatype)
        output-buf-a (drv/allocate-host-buffer device 10 datatype)
        buf-b (drv/allocate-device-buffer device 10 datatype)
        input-data (dtype/make-array-of-type datatype (range 10))
        output-data (dtype/make-array-of-type datatype 10)]
    (dtype/copy! input-data 0 buf-a 0 10)
    (dtype/set-value! buf-a 0 100.0)
    (dtype/copy! buf-a 0 output-data 0 10)
    (drv/copy-host->device stream buf-a 0 buf-b 0 10)
    (drv/memset stream buf-b 5 20.0 5)
    (drv/copy-device->host stream buf-b 0 output-buf-a 0 10)
    (drv/wait-for-event (drv/create-event stream))
    (dtype/copy! output-buf-a 0 output-data 0 10)
    (is (= [100.0 1.0 2.0 3.0 4.0 20.0 20.0 20.0 20.0 20.0]
           (mapv double output-data)))))

(defn create-make-buffer
  [device stream datatype]
   (fn [elem-seq]
     (drv/host-array->device-buffer device stream
                                    (dtype/make-array-of-type datatype elem-seq))))

(defmacro backend-test-pre
  [device datatype & body]
  `(let [~'stream (drv/create-stream ~device)
         ~'make-buffer (create-make-buffer ~device ~'stream ~datatype)
         ~'->array (fn [buffer#] (drv/device-buffer->host-array ~device ~'stream buffer#))]
     ~@body
     ))

(defn gemm
  [device datatype]
  (backend-test-pre device datatype
   (let [buf-a (make-buffer (flatten (repeat 4 (range 10))))
         buf-b (make-buffer (flatten (repeat 10 (range 5))))
         buf-c (make-buffer (repeat 20 0))
         a-row-count 4
         a-col-count 10
         b-row-count a-col-count
         b-col-count 5
         c-row-count a-row-count
         c-col-count b-col-count]
     (math/gemm stream false false
                a-row-count a-col-count b-row-count b-col-count c-row-count c-col-count
                1.0 buf-a a-col-count
                buf-b b-col-count
                0.0 buf-c c-col-count)
     (is (= (vec (map double (flatten (repeat 4 [0 45 90 135 180]))))
            (vec (map double (->array buf-c))))))))

(defn sum
  [device datatype]
  (backend-test-pre device datatype
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
  [device datatype]
  (backend-test-pre device datatype
   (let [values (flatten (repeat 4 (range 10)))
         buf-a (make-buffer values)
         buf-b (make-buffer values)
         result (make-buffer (repeat (count values) 0))
         answer (m/sub values (m/mul values 2))]
     (math/subtract stream 1.0 buf-a 2.0 buf-b result)
     (is (= (vec (map double answer))
            (vec (map double (->array result))))))))

(defn gemv
  [device datatype]
  (backend-test-pre device datatype
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
  [device datatype]
  (backend-test-pre device datatype
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
  [device datatype]
  (backend-test-pre device datatype
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
  [device datatype]
  (backend-test-pre device datatype
   (let [values (mapv #(* % %) (range 10))
         buf-a (make-buffer values)
         answer [1 1 1 1 1 1 (/ 5 6) (/ 5 7) (/ 5 8) (/ 5 9)]]
     (math/l2-constraint-scale stream buf-a 1 5)
     (is (close-enough (mapv double answer)
                       (mapv double (->array buf-a)))))))
