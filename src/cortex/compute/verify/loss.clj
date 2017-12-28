(ns cortex.compute.verify.loss
  (:require [clojure.test :refer :all]
            [clojure.core.matrix :as m]
            [cortex.compute.nn.backend :as backend]
            [cortex.compute.driver :as drv]
            [cortex.compute.math :as math]
            [cortex.graph :as graph]
            [cortex.nn.layers :as layers]
            [cortex.nn.network :as network]
            [cortex.loss.util :as util]
            [cortex.loss.core :as cortex-loss]
            [clojure.core.matrix :as m]))


(defn- normalize
  [radius vec-data]
  (let [mag-data (Math/sqrt
                  (double
                   (->> vec-data
                        (map #(* (double %)
                                 (double %)))
                        (reduce +))))]
    (m/mul vec-data (/ radius mag-data))))


(defn center-loss
  [backend]
  (backend/with-backend backend
   (let [n-classes 5
         n-features n-classes
         batch-size 6
         center-val 1.0
         feature-val 1.0
         alpha 0.5
         radius 10.0

         input-buffer-map {:labels [[1 0 0 0 0]
                                    [0 0 1 0 0]
                                    [1 0 0 0 0]
                                    [0 0 0 1 0]
                                    [0 0 0 1 0]
                                    [1 0 0 0 0]]}
         centers (vec (repeat n-classes (vec (repeat n-features center-val))))
         features (mapv #(m/mul % 20) (:labels input-buffer-map))
         gradients (mapv #(m/sub % 1) features)
         network (network/linear-network [(layers/input n-features 1 1)
                                          (layers/linear n-features :id :feature)])

         graph (-> (graph/add-node (network/network->graph network)
                                   (cortex-loss/center-loss :labels {:stream :labels}
                                                            :output {:node-id :feature
                                                                     :type :node-output}
                                                            :label-indexes {:stream :labels}
                                                            :label-inverse-counts {:stream :labels}
                                                            :radius radius
                                                            :alpha alpha
                                                            :centers {:buffer centers}
                                                            :id :center-loss-1)
                                   [:feature])
                   first
                   (graph/add-stream :labels (graph/stream-descriptor 5))
                   graph/build-graph
                   graph/generate-parameters)
         stream (backend/get-stream)
         stream->buffer-map (->> (graph/augment-streams graph input-buffer-map)
                                 (map (fn [[k v]]
                                        (if (map? v)
                                          [k {:buffer (math/array stream (get v :datatype) (get v :data))}]
                                          [k {:buffer (backend/array backend v)}])))
                                 (into {}))
         ;;Upload buffers to device.
         graph (update graph :buffers (fn [buffer-map]
                                        (->> buffer-map
                                             (map (fn [[k v]]
                                                    [k (update v :buffer #(backend/array backend %))]))
                                             (into {}))))
         loss-term (graph/get-node graph :center-loss-1)
         argument-map (graph/resolve-arguments graph (graph/get-node graph :center-loss-1)
                                               stream->buffer-map
                                               {:feature {:buffer (backend/array backend features batch-size)
                                                          :gradient (backend/new-array backend [n-features]
                                                                                       batch-size)}})
         loss-term (util/create-compute-loss-term backend {:compute-graph graph} loss-term batch-size)
         adjusted-centers (->> [1 0 1 1 0]
                               (map-indexed (fn [idx nonzero?]
                                              (->> (if (= 1 nonzero?)
                                                     (assoc (vec (repeat n-features 0.5))
                                                            idx 10.5)
                                                     (vec (repeat n-features 1)))
                                                   (normalize radius))))
                               vec)]
     (util/compute-loss-gradient loss-term argument-map)
     (let [output-gradients (->> (backend/to-double-array backend (get-in argument-map [:output :gradient]))
                                 (partition n-features)
                                 (mapv vec))]
       (is (m/equals gradients
                     output-gradients))
       (is (m/equals adjusted-centers
                     (->> (backend/to-double-array backend (get-in argument-map [:centers :buffer]))
                          (partition n-features)
                          (mapv vec))
                     1e-4))))))
