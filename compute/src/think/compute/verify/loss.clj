(ns think.compute.verify.loss
  (:require [clojure.test :refer :all]
            [think.compute.loss :as compute-loss]
            [cortex.loss :as cortex-loss]
            [think.compute.nn.backend :as backend]
            [clojure.core.matrix :as m]
            [think.compute.driver :as drv]
            [think.compute.math :as math]))



(defn center-loss
  [backend]
  (let [n-classes 5
        n-features 10
        batch-size 6
        center-val 1.0
        feature-val 1.0
        alpha 0.5
        input-buffer-map {:labels [[1 0 0 0 0]
                                   [0 0 1 0 0]
                                   [1 0 0 0 0]
                                   [0 0 0 1 0]
                                   [0 0 0 1 0]
                                   [1 0 0 0 0]]}
        centers (vec (repeat n-classes (vec (repeat n-features center-val))))
        features (vec (repeatedly batch-size #(vec (repeat n-features feature-val))))
        gradients (mapv #(m/sub % 1) features)
        loss-fn (->> [(cortex-loss/center-loss :labels {:stream :labels}
                                               :output {:node-id :feature
                                                        :type :node-output}
                                               :alpha alpha)]
                     cortex-loss/generate-augmented-argument-ids)
        driver (drv/get-driver backend)
        stream (drv/get-stream backend)
        stream-buffer-map (->> (cortex-loss/augment-streams loss-fn input-buffer-map)
                               (map (fn [[k v]]
                                      (if (map? v)
                                        [k (math/array driver stream (get v :datatype) (get v :data))]
                                        [k (backend/array backend v)])))
                               (into {}))
        node-id->name->shape-map {:feature {:output [n-features]}}
        loss-term (first loss-fn)
        argument-map (->> (concat (cortex-loss/get-loss-term-augmented-streams loss-term)
                                  (cortex-loss/get-loss-term-streams loss-term))
                          (map (fn [{:keys [key type id]}]
                                 (condp = type
                                   :stream
                                   [key {:buffer (get stream-buffer-map key)}]
                                   :stream-augmentation
                                   [key {:buffer (get stream-buffer-map id)}])))
                          (into {})
                          (merge {:output {:buffer (backend/array backend features batch-size)
                                           :gradient (backend/new-array backend [n-features] batch-size)}
                                  :centers {:buffer (backend/array backend centers)}}))
        loss-term (compute-loss/create-compute-loss-term (first loss-fn) backend node-id->name->shape-map
                                                         {:labels n-classes}
                                                         batch-size)
        nonzero-classes [1 0 1 1 0]
        adjusted-centers (mapv #(if (zero? %)
                                  (vec (repeat n-features 1.0))
                                  (vec (repeat n-features (+ (* alpha center-val)
                                                             (* (- 1.0 alpha) feature-val)))))
                               nonzero-classes)]
    (compute-loss/compute-loss-gradient loss-term argument-map)
    (let [output-gradients (->> (backend/to-double-array backend (get-in argument-map [:output :gradient]))
                                (partition n-features)
                                (mapv vec))]
      (is (m/equals gradients
                    output-gradients))
      (is (m/equals adjusted-centers
                    (->> (backend/to-double-array backend (get-in argument-map [:centers :buffer]))
                         (partition n-features)
                         (mapv vec)))))))
