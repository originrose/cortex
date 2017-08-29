(ns cortex.compute.cpu-optimize-test
  (:require
   [cortex.compute.cpu.backend :as cpu-backend]
   [cortex.compute.verify.optimize :as verify-optimize]
   [clojure.test :refer :all]
   [cortex.compute.verify.utils :refer [def-double-float-test] :as verify-utils]))

(use-fixtures :each verify-utils/test-wrapper)

(defn create-backend
  []
  (cpu-backend/backend :datatype verify-utils/*datatype*))

(def-double-float-test adam
  (verify-optimize/test-adam (create-backend)))
