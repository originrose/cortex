(ns cortex.optimise.optimisers
  "Contains protocol extensions for gradient optimisers, as
  well as a selection of sample gradient optimisers for use
  in optimizing pure functions or training neural networks.

  The gradient optimisers usable with the functions in
  cortex.optimise.descent implement the following protocols:

  PParameters - to allow for retrieving an updated parameter vector
  PGradientOptimiser - to allow for passing in parameter and gradient vectors
  PIntrospection - to allow for inspecting the internal state of the optimiser

  In this namespace, the above protocols are extended to Clojure
  maps and Clojure functions. See cortex.optimise.parameters for
  the reason that APersistentMap rather than IPersistentMap is
  used.

  (Note that the PParameters protocol is also implemented by
  pure functions, so it is not done here, but rather in the
  shared namespace cortex.optimise.parameters.)

  A Clojure function representing a gradient optimiser must
  take parameter and gradient vectors and return an updated
  parameter vector.

  A Clojure map representing a gradient optimiser must have
  the two keys :initialize and :update, which correspond to
  functions. The :initialize function should take a parameter
  count and return a state map (with, for instance, any accumulation
  vectors initialized to the correct sizes). The :update function
  should take the state map (with the parameter vector under
  the :params key) and a gradient vector, and return a new
  state map with the updated parameter vector under the
  :params key."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :refer [+ - * /]] [cortex.nn.protocols :as cp]
            [cortex.optimise.parameters]))

;;;; Protocol extensions

(extend-type clojure.lang.IFn
  cp/PGradientOptimiser
  (compute-parameters [this gradient parameters]
    (cp/compute-parameters
      {:update (fn [state gradient]
                 (let [state (update state :params this gradient)]
                   ;; This makes it much easier to debug a common mistake:
                   (when (map? (:params state))
                     (throw (IllegalStateException.
                              "fn acting as optimiser must return vector: did you need to call the fn to produce an optimiser map or fn?")))
                   state))}
      gradient
      parameters)))

(extend-type clojure.lang.APersistentMap
  cp/PGradientOptimiser
  (compute-parameters [this gradient parameters]
    (as-> this this
      (if (:initialize this)
        (-> this
          (assoc :state ((:initialize this) (m/ecount parameters)))
          (dissoc :initialize))
        this)
      (-> this
        (assoc-in [:state :params] parameters)
        (update :state (:update this) gradient))))

  cp/PIntrospection
  (get-state [this]
    (dissoc (:state this) :params)))

;;;; Clojure implementations

(defn sgd-clojure
  [& {:keys [learning-rate]
      :or {learning-rate 0.1}}]
  (fn [params gradient]
    (+ params
       (* (- learning-rate)
          gradient))))

(defn accumulate
  [decay-rate running-avg value]
  (+ (* decay-rate running-avg)
     (* (- 1 decay-rate) value)))

(defn adadelta-clojure
  [& {:keys [decay conditioning]
      :or {decay 0.95
           conditioning 1e-6}}]
  (letfn [(acc [acc-x x]
            (accumulate decay acc-x x))
          (rms [acc-x]
            (m/sqrt
              (+ acc-x
                 conditioning)))]
    {:initialize (fn [param-count]
                   {:acc-gradient (m/new-vector :vectorz param-count)
                    :acc-step (m/new-vector :vectorz param-count)})
     :update (fn [{:keys [params acc-gradient acc-step]} gradient]
               (let [acc-gradient (acc acc-gradient (m/square gradient))
                     step (-> gradient
                            (* (rms acc-step))
                            (/ (rms acc-gradient)))
                     acc-step (acc acc-step (m/square step))
                     params (- params step)]
                 {:acc-gradient acc-gradient
                  :acc-step acc-step
                  :params params}))}))

(defn adam-clojure
  [& {:keys [step-size first-moment-decay
             second-moment-decay conditioning]
      :or {step-size 0.001
           first-moment-decay 0.9
           second-moment-decay 0.999
           conditioning 1e-8}}]
  {:initialize (fn [param-count]
                 {:first-moment (m/new-vector :vectorz param-count)
                  :second-moment (m/new-vector :vectorz param-count)
                  :num-steps 0})
   :update (fn [{:keys [params first-moment second-moment num-steps]} gradient]
             (let [num-steps (inc num-steps)
                   first-moment (accumulate first-moment-decay
                                            first-moment
                                            gradient)
                   second-moment (accumulate second-moment-decay
                                             second-moment
                                             (m/square gradient))
                   first-moment* (/ first-moment
                                    (- 1 (Math/pow first-moment-decay num-steps)))
                   second-moment* (/ second-moment
                                     (- 1 (Math/pow second-moment-decay num-steps)))
                   step (* step-size
                           (/ first-moment*
                              (+ (m/sqrt second-moment*)
                                 conditioning)))
                   params (- params step)]
               {:params params
                :first-moment first-moment
                :second-moment second-moment
                :num-steps num-steps}))})
