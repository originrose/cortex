(ns think.compute.cuda-optimise-test
  (:require [think.compute.nn.cuda-backend :as cuda-backend]
            [think.compute.verify.optimise :as verify-optimise]
            [clojure.test :refer :all]
            [think.compute.verify.utils :as verify-utils]))

(use-fixtures :each verify-utils/test-wrapper)


(defn create-backend
  []
  (cuda-backend/create-backend verify-utils/*datatype*))


(verify-utils/def-double-float-test adam
  (verify-optimise/test-adam (create-backend)))
