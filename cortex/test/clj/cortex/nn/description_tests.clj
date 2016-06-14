(ns cortex.nn.description-tests
  (:require
    [clojure.test :refer [deftest is are]]
    [cortex.nn.network :as net]
    [cortex.nn.description :as desc]))


(defn create-mnist-network
  "This test will fail if regression occurs against invalid bias when bias
  should be nil."
  []
  (let [network-desc [(desc/input 28 28 1)
                      (desc/convolutional 5 0 1 20)
                      (desc/max-pooling 2 0 2)
                      (desc/convolutional 5 0 1 50)
                      (desc/max-pooling 2 0 2)
                      (desc/linear->relu 500)
                      (desc/linear->softmax 10)]
        built-network (desc/build-full-network-description network-desc)]
    (desc/create-network built-network)))

(deftest build-mnist-sample
  (create-mnist-network))
