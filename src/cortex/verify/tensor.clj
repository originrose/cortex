(ns cortex.verify.tensor
  (:require [cortex.tensor :as ct]
            [cortex.compute.cpu.driver :as cpu-driver]
            [cortex.compute.driver :as drv]
            [clojure.test :refer :all]
            [clojure.core.matrix :as m]))


(deftest assign-constant!
  []
  (let [driver (cpu-driver/driver)
        stream (drv/create-stream driver)]
    (with-bindings {#'ct/*stream* stream}
      (let [tensor (ct/->tensor (partition 3 (range 9)))]
        (is (= (ct/ecount tensor) 9))
        (is (m/equals (range 9)
                      (ct/to-double-array tensor)))))))
