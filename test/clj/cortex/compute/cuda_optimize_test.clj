(ns ^:gpu cortex.compute.cuda-optimize-test
  (:require [cortex.compute.verify.optimize :as verify-optimize]
            [clojure.test :refer :all]
            [cortex.compute.verify.utils :as verify-utils]))

(use-fixtures :each verify-utils/test-wrapper)

(defn create-backend
  []
  (require '[cortex.compute.cuda.backend :as cuda-backend])
  ((resolve 'cuda-backend/backend) :datatype verify-utils/*datatype*))


(verify-utils/def-double-float-test adam
  (verify-optimize/test-adam (create-backend)))
