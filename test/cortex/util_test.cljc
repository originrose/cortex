(ns cortex.util-test
  (:require #?(:cljs
                [cljs.test :refer-macros [deftest is testing]]
                :clj
                [clojure.test :refer [deftest is testing]])
            [cortex.util :as util]))


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
