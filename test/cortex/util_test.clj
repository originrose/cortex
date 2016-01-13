(ns cortex.util-test
  (:require [cortex.util :as util]
            [clojure.test :refer [deftest is are]]))


(deftest confusion-test
  (let [cf (util/confusion-matrix ["cat" "dog" "rabbit"])
        cf (-> cf
            (util/add-prediction "dog" "cat")
            (util/add-prediction "dog" "cat")
            (util/add-prediction "cat" "cat")
            (util/add-prediction "cat" "cat")
            (util/add-prediction "rabbit" "cat")
            (util/add-prediction "dog" "dog")
            (util/add-prediction "cat" "dog")
            (util/add-prediction "rabbit" "rabbit")
            (util/add-prediction "cat" "rabbit")
            )]
    (util/print-confusion-matrix cf)
    (is (= 2 (get-in cf ["cat" "dog"])))))
