(ns cortex.optimise.descent
  "Contains API functions for performing gradient descent on pure
  functions using gradient optimisers."
  (:refer-clojure :exclude [+ - * /])
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :refer [+ - * /]]
            [clojure.string :as str]
            [cortex.nn.protocols :as cp]
            [cortex.optimise.functions]
            [cortex.optimise.optimisers]
            [cortex.util :as util]))

;;;; Config

(defonce ^:dynamic *emacs* false)
(defn emacs [] (alter-var-root #'*emacs* (constantly true)))
(defn no-emacs [] (alter-var-root #'*emacs* (constantly false)))

;;;; Basic gradient descent

(defn do-steps
  "Performs num-steps iterations of gradient descent, printing
  the parameters, function value, and gradient optimiser state
  at each stage."
  [function optimiser initial-params num-steps]
  (loop [params initial-params
         optimiser optimiser
         step-count 0]
    (let [function (cp/update-parameters function params)
          value (cp/output function)
          gradient (cp/gradient function)]
      (print (str "f" (vec params) " = " value "; state = " ))
      (if (< step-count num-steps)
        (let [optimiser (cp/compute-parameters optimiser gradient params)
              params (cp/parameters optimiser)
              state (cp/get-state optimiser)]
          (println state)
          (recur params
                 optimiser
                 (inc step-count)))
        (println "(done)")))))

;;;; Interactive gradient descent

(defn interactive
  [function initial-params
   & {:keys [normalize? prompt]
      :or {normalize? false
           prompt "f = %1$ .5e, |grad| = %2$ .5e, learning rate = "}}]
  (let [function (cp/update-parameters function initial-params)]
    (loop [lrate-history []
           params-history [initial-params]
           value-history [(cp/output function)]
           gradient-history [(cp/gradient function)]]
      (if-let [{:keys [lrate force? back]}
               (loop []
                 (print (util/sformat prompt
                                      (peek value-history)
                                      (m/magnitude (peek gradient-history))))
                 (flush)
                 (let [input (read-line)
                       _ (when *emacs* (println input))
                       input (str/trim input)]
                   (if (.startsWith input "back")
                     (let [input (-> input
                                   (str/replace-first #"^back" "")
                                   (str/trim))]
                       (or (try
                             {:back (if (seq input)
                                      (let [back (util/parse-long input)]
                                        (if (neg? back)
                                          (throw (NumberFormatException.))
                                          back))
                                      1)}
                             (catch NumberFormatException _))
                           (recur)))
                     (let [force? (.startsWith input "force")
                           input (-> input
                                   (str/replace-first #"^force" "")
                                   (str/trim))]
                       (if (seq input)
                         (or (try
                               {:lrate (util/parse-double input)
                                :force? force?}
                               (catch NumberFormatException _))
                             (when force? (recur)))
                         (if-let [last-lrate (peek lrate-history)]
                           {:lrate last-lrate
                            :force? force?}
                           (recur)))))))]
        (if back
          (let [length (max 1 (- (count params-history) back))]
            (recur (m/subvector lrate-history 0 (dec length))
                   (m/subvector params-history 0 length)
                   (m/subvector value-history 0 length)
                   (m/subvector gradient-history 0 length)))
          (let [step (* (if normalize?
                          (m/normalise (peek gradient-history))
                          (peek gradient-history))
                        (- lrate))
                params (+ (peek params-history)
                          step)
                function (cp/update-parameters function params)
                value (cp/output function)]
            (if (or (< value (peek value-history))
                    force?)
              (recur (conj lrate-history lrate)
                     (conj params-history params)
                     (conj value-history value)
                     (conj gradient-history (cp/gradient function)))
              (recur (conj lrate-history lrate)
                     (conj params-history (peek params-history))
                     (conj value-history (peek value-history))
                     (conj gradient-history (peek gradient-history))))))
        (peek params-history)))))

;;;; Instrumented gradient descent

(defn study
  [function optimiser initial-params terminate?]
  (loop [last-params nil
         params initial-params
         optimiser optimiser
         initial-step true]
    (let [function (cp/update-parameters function params)
          gradient (cp/gradient function)
          state (util/->lazy-map
                  (merge
                    (if-not initial-step
                      (merge
                        ;; This *needs* to be first, because lazy maps will be coerced
                        ;; to regular maps if they appear in the tail of a merge, and
                        ;; the map returned by get-state might be lazy.
                        (cp/get-state optimiser)
                        {:last-step (delay (- params last-params))}))
                    {:params params
                     :value (delay (cp/output function))
                     :gradient gradient
                     :optimiser optimiser
                     :initial-step initial-step}))]
      (if-not (terminate? state)
        (let [optimiser (cp/compute-parameters optimiser gradient params)
              new-params (cp/parameters optimiser)]
          (recur params
                 new-params
                 optimiser
                 false))
        params))))

(def do-study
  (comp (constantly nil)
        study))

(defn log
  [function optimiser initial-params terminate?]
  (let [log* (atom [])]
    (study
      function
      optimiser
      initial-params
      (fn [state]
        (swap! log* conj (util/force-map state))
        (terminate? state)))
    @log*))

(defn review
  [terminate? & logs]
  (loop [entry-vectors (apply map vector logs)]
    (when-not (or (empty? entry-vectors)
                  (terminate? (first entry-vectors)))
      (recur (rest entry-vectors)))))

;;;; Termination function builders

(defmacro term
  [& transforms]
  `(-> (constantly false) ~@(reverse transforms)))

;;;; Higher-order termination functions

(defn skip-initial
  ([terminate?]
   (skip-initial terminate? 1))
  ([terminate? n]
   (let [num-calls* (atom 0)]
     (fn [state]
       (when (> (swap! num-calls* inc) n)
         (terminate? state))))))

(defn stagger
  [terminate?]
  (let [last-state* (atom nil)]
    (fn [state]
      (let [last-state @last-state*]
        (terminate?
          {:last last-state
           :current (reset! last-state* state)})))))

(defn count-steps
  [terminate?]
  (let [num-steps* (atom -1)]
    (fn [state]
      (terminate? (assoc state :step-count (swap! num-steps* inc))))))

;;;; Termination conditions

(defn limit-steps
  [terminate? n]
  (let [num-steps* (atom 0)]
    (fn [state]
      (or (terminate? state)
          (> (swap! num-steps* inc) n)))))

(defn limit-params
  [terminate? cutoff params]
  (fn [state]
    (or (terminate? state)
        (<= (m/distance (- (:params state)
                           params))
            cutoff))))

(defn limit-value
  [terminate? cutoff]
  (fn [state]
    (or (terminate? state)
        (<= (:value state)
            cutoff))))

(defn limit-gradient
  [terminate? cutoff]
  (fn [state]
    (or (terminate? state)
        (<= (m/magnitude (:gradient state))
            cutoff))))

;;;; I/O

(defn slow-steps
  [terminate? msecs]
  (fn [state]
    (when-not (terminate? state)
      (Thread/sleep msecs))))

(defn outputf
  [terminate? fmt & args]
  (fn [state]
    (apply util/sprintf fmt
           (for [arg args]
             (let [transforms (if (vector? arg)
                                arg
                                [arg])]
               (loop [state state
                      transforms transforms]
                 (if (seq transforms)
                   (let [transform (first transforms)]
                     (recur (cond
                              (fn? transform)
                              (transform state)
                              (associative? state)
                              (get state transform))
                            (rest transforms)))
                   state)))))
    (flush)
    (terminate? state)))

(defn output
  [terminate? & args]
  (apply outputf terminate?
         (str (str/join \space
                        (repeat (count args) "%s"))
              "%n")
         args))

(defn stdout
  [terminate?]
  (outputf terminate?
           "f = % .8e, |grad| = % .8e, |dx| = % .8e%n"
           :value
           (comp m/magnitude :gradient)
           (fn [state]
             (if-not (:initial-step state)
               (m/magnitude (:last-step state))
               Double/NaN))))

(defn pause
  ([terminate?] (pause terminate? 1))
  ([terminate? period]
   (let [step-count* (atom -1)
         period* (atom period)]
     (fn [state]
       (or (terminate? state)
           (when (>= (swap! step-count* inc) @period*)
             (let [input (read-line)]
               (when *emacs* (println input))
               (reset! step-count* 0)
               (when (seq input)
                 (try
                   (reset! period* (util/parse-long input))
                   false
                   (catch NumberFormatException _ true))))))))))
