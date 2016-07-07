(ns cortex.optimize.optimizers
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :refer [+ - * /]]
            [cortex.optimize.protocols :as P]))

(extend-protocol P/Optimizer
  clojure.lang.IFn
  (initialize [this param-count]
    {:update (fn [state gradient]
               (assoc state :step
                      (this gradient)))
     :state {}})
  ;; This turns the Optimizer into a map, so we don't need to
  ;; implement any of the other methods.

  clojure.lang.IPersistentMap
  (initialize [this param-count]
    (-> this
      (assoc :state ((:initialize this) param-count))
      (dissoc :initialize)))
  (compute-update [this gradient] (update this :state (:update this) gradient))
  (get-step [this] (:step (:state this))))

(defn sgd
  [& {:keys [learning-rate]
      :or {learning-rate 0.1}}]
  (fn [gradient]
    (* (- learning-rate)
       gradient)))

(defn adadelta
  [& {:keys [rho epsilon]
      :or {rho 0.95
           epsilon 1e-6}}]
  (letfn [(acc [acc-x x]
            (+ (* rho acc-x)
               (* (- 1 rho) x)))
          (rms [acc-x]
            (m/sqrt
              (+ acc-x
                 epsilon)))]
    {:initialize (fn [param-count]
                   {:acc-gradient (m/new-vector :vectorz param-count)
                    :acc-step (m/new-vector :vectorz param-count)})
     :update (fn [{:keys [acc-gradient acc-step]} gradient]
               (let [acc-gradient (acc acc-gradient (m/square gradient))
                     step (-> gradient
                            (* (rms acc-step))
                            (/ (rms acc-gradient))
                            -)
                     acc-step (acc acc-step (m/square step))]
                 {:acc-gradient acc-gradient
                  :acc-step acc-step
                  :step step}))}))
