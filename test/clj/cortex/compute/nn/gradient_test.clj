(ns cortex.compute.nn.gradient-test
  (:require [clojure.test :refer :all]
            [cortex.verify.nn.gradient :as gradient]
            [cortex.compute.cpu.backend :as cpu-backend]
            [cortex.compute.verify.utils :as test-utils]
            [cortex.nn.execute :as execute]))


(defn create-context
  []
  (execute/compute-context :backend :cpu
                           :datatype test-utils/*datatype*))


(deftest corn-gradient
  (gradient/corn-gradient (create-context)))

(deftest batch-normalization
  (gradient/batch-normalization-gradient (create-context)))

;; (deftest local-response-normalization-gradient
;;   (gradient/lrn-gradient (create-context)))

(deftest prelu-gradient
  (gradient/prelu-gradient (create-context)))

(deftest concat-gradient
  (gradient/concat-gradient (create-context)))

(deftest split-gradient
  (gradient/split-gradient (create-context)))

(deftest join-+-gradient
  (gradient/join-+-gradient (create-context)))

(deftest join-*-gradient
  (gradient/join-*-gradient (create-context)))

(deftest censor-gradient
  (gradient/censor-gradient (create-context)))

(deftest yolo-gradient
  (gradient/yolo-gradient (create-context)))
