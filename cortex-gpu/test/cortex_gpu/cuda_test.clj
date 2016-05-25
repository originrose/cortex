(ns cortex-gpu.cuda-test
  (:require [clojure.test :refer :all]
            [cortex-gpu.test-framework :as framework]
            [cortex-gpu.cuda :as cuda]
            [clojure.java.io :as io]))


(use-fixtures :once framework/with-contexts)
(use-fixtures :each framework/with-resource-context)


(deftest vecadd-cuda
  (let [module (cuda/load-module (io/input-stream (io/resource "vectorAdd_kernel.fatbin")))
        vecadd-fn (cuda/get-function module "VecAdd_kernel")
        item-count 100000
        item-size (* item-count 8)
        vec-a-data (range 1 (inc item-count))
        vec-b-data (repeat item-count 40)
        dev-a (cuda/mem-alloc item-size)
        dev-b (cuda/mem-alloc item-size)
        dev-c (cuda/mem-alloc item-size)
        _ (cuda/mem-copy-host->device (cuda/doubles-to-ptr vec-a-data) dev-a item-size)
        _ (cuda/mem-copy-host->device (cuda/doubles-to-ptr vec-b-data) dev-b item-size)
        threads-per-block 256
        blocks-per-grid (/ (+ item-count (- threads-per-block 1)) threads-per-block)]
    (cuda/launch-kernel vecadd-fn
                        blocks-per-grid 1 1
                        threads-per-block 1 1
                        0
                        dev-a dev-b dev-c item-count)
    (cuda/check-errors)
    (is (= (map double (range 41 (+ 41 20)))
           (take 20 (seq (cuda/mem-copy-doubles-device->host dev-c item-size)))))))
