(ns cortex.compute.nn.train-test
  (:require [clojure.test :refer :all]
            [cortex.verify.nn.train :as verify-train]
            [cortex.compute.cpu.backend :as cpu-backend]
            [cortex.nn.execute :as execute]
            [cortex.compute.verify.utils
             :refer [def-double-float-test]
             :as test-utils]))


(use-fixtures :each test-utils/test-wrapper)


(defn create-context
  []
  (execute/compute-context :backend :cuda
                           :datatype test-utils/*datatype*))


(def-double-float-test corn
  (verify-train/test-corn (create-context)))


(def-double-float-test mnist
  (verify-train/train-mnist (create-context)))

