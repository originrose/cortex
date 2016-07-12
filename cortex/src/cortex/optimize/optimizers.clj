(ns cortex.optimize.optimizers
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :refer [+ - * /]]
            [cortex.optimize.protocols :as P]))

;;; The Optimizer protocol is extended to the following two types:
;;;
;;; A map with keys :initialize and :update, the first of which
;;; should correspond to a function that will take the initial-params
;;; and return a state map, and the second of which should correspond
;;; to a function that will take the state map, with the initial-params
;;; assoc'd onto the :params key, and return a new state map with the
;;; :params key and any other keys necessary updated.
;;;
;;; A function that will take the params and gradient and return a new
;;; set of params. Note that this is appropriate only for stateless
;;; optimizers like SGD.

(extend-protocol P/Optimizer
  clojure.lang.IFn
  (initialize [this initial-params]
    {:update (fn [state gradient]
               (update state :params this gradient))
     :state {:params initial-params}})
  ;; This turns the Optimizer into a map, so it doesn't make
  ;; sense to implement any of the other methods.

  clojure.lang.IPersistentMap
  (initialize [this initial-params]
    (-> this
      (assoc :state ((:initialize this) initial-params))
      (assoc-in [:state :params] initial-params)
      (dissoc :initialize)))
  (get-params [this]
    (get-in this [:state :params]))
  (update-params [this gradient]
    (update this :state (:update this) gradient))
  (get-state [this]
    (:state this)))

(defn sgd
  [& {:keys [learning-rate]
      :or {learning-rate 0.1}}]
  (fn [params gradient]
    (+ params
       (* (- learning-rate)
          gradient))))

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
    {:initialize (fn [initial-params]
                   {:acc-gradient (m/new-vector :vectorz (count initial-params))
                    :acc-step (m/new-vector :vectorz (count initial-params))})
     :update (fn [{:keys [params acc-gradient acc-step]} gradient]
               (let [acc-gradient (acc acc-gradient (m/square gradient))
                     step (-> gradient
                            (* (rms acc-step))
                            (/ (rms acc-gradient))
                            -)
                     acc-step (acc acc-step (m/square step))]
                 {:acc-gradient acc-gradient
                  :acc-step acc-step
                  :params (+ params step)}))}))
