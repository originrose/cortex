(ns cortex.optimise.descent
  "Contains API functions for performing gradient descent on pure
  functions using gradient optimisers."
  (:refer-clojure :exclude [+ - * / defonce])
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.operators :refer [+ - * /]]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [cortex.nn.protocols :as cp]
            [cortex.optimise.functions :as function]
            [cortex.optimise.optimisers :as opts]
            [cortex.util :as util :refer [defonce]]))

;;;; Config

(defonce ^:dynamic *emacs*
  "In the Emacs REPL, user input read from stdin is not printed to
  the console. This leads to a very confusing and unreadable output
  for interactive sessions (such as when running the 'interactive'
  gradient descent function), so you can turn on stdin echoing by
  toggling *emacs* to a truthy value."
  false)
(defn emacs
  "Turn on stdin echoing support (do this if you use Emacs)."
  []
  (alter-var-root #'*emacs* (constantly true)))
(defn no-emacs
  "Turn off stdin echoing support, if it has been turned on."
  []
  (alter-var-root #'*emacs* (constantly false)))

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
  "Performs interactive gradient descent. The interactive part is
  that before each step, you will be prompted for the learning rate.
  (If :normalize? is falsy, though, the gradient vector will be
  normalized before multiplying it by the learning rate -- so in this
  case, you are specifying the step size rather than the learning rate.)
  You can enter the learning rate in any format accepted by
  Double/parseDouble, and it can even be negative (although you would
  then be performing gradient *ascent* rather than descent). If you
  don't specify a learning rate, the last one you entered is used again.
  (But you cannot do this on the first step, because there is not any
  previous entry at that point.) By default, a step will be aborted if
  it would increase the value of the objective function, but you can
  override this behavior by starting your entry with 'force' (as in
  'force 1', or just 'force' -- the latter version would reuse your
  last entered learning rate). A complete history of past steps is
  kept, and every input you make (even if the step is aborted)
  corresponds to an entry in the history. You can revert to previous
  entries in the history by typing 'back 5' (to go back five entries)
  or just 'back' (to go back one entry). Note that the 'previously
  entered step' is determined by the history, so if you go to a previous
  state in the history and then take a step without providing a learning
  rate, the learning rate used will be the last one entered *before that
  point in the history*, not the last one that you literally entered
  in real life. You can customize the prompt by passing a new string for
  :prompt; this string will be passed to util/sformat with the function
  value and gradient as positional arguments."
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
  "Performs gradient descent on the given function using the provided
  optimiser, starting from the specified initial parameter vector,
  until terminate? returns true. Before each step (even the first
  one), terminate? is called with a lazy map containing :params (the
  current parameter vector), :value (the current value of the
  objective function), :gradient (the current gradient of the
  objective function), :optimiser (the current optimiser object),
  and :initial-step (whether or not at least one step has actually
  been performed). On every step but the initial step, this map will
  also contain :last-step (the last step taken, as a vector) and
  anything from the map returned by calling get-state on the
  optimiser.  Returns the parameter vector at the first time the
  terminate? function returns truthy."
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
  "Like study, but returns nil."
  (comp (constantly nil)
        study))

(defn log
  "Like study, but returns a vector of the maps that were passed to
  the terminate? function. Each map is fully realized (i.e. it is made
  from a lazy map into a regular map)."
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
  "Like study, but instead of actually performing gradient descent,
  it just reads from one or more logs to get the maps to pass to the
  terminate? function. If only one log is passed, each map is passed
  in turn. If more than one log is passed, successive vectors of log
  entries (which are maps) are passed in turn. To access data in this
  case, the terminate? function will have to index into the vector,
  then access one of the keys in the map. Passing multiple logs is a
  nice way to view more than one algorithm (or a single algorithm
  with different hyperparameters) side by side."
  [terminate? & logs]
  (loop [entry-vectors (apply map vector logs)]
    (when-not (or (empty? entry-vectors)
                  (terminate? (if (> (count logs) 1)
                                (first entry-vectors)
                                (ffirst entry-vectors))))
      (recur (rest entry-vectors)))))

(defn run-test
  "Run a predefined gradient descent test. Effectively passes the args
  in the second argument to study, with some special functionality
  depending on mode. If mode is :profile, returns a map containing the
  number of times the function's value was calculated (:f), the number
  of times the function's gradient was calculated (:grad), and the
  total CPU time required for the gradient descent (:time, in
  milliseconds). If mode is :study, :do-study, or :log, returns like
  the respective functions. Passing another mode is an error."
  [mode {:keys [function optimiser initial-params terminate?]}]
  (case mode
    :profile
    (let [value* (atom 0)
          gradient* (atom 0)]
      (-> (util/ctime*
            (study
              {:value (fn [params]
                        (swap! value* inc)
                        (function/value function params))
               :gradient (fn [params]
                           (swap! gradient* inc)
                           (function/gradient function params))}
              optimiser
              initial-params
              terminate?))
        (dissoc :return)
        (assoc :value @value*)
        (assoc :gradient @gradient*)))
    (:study :do-study :log)
    (apply (case mode
             :study study
             :do-study do-study
             :log log)
           function
           optimiser
           initial-params
           terminate?)
    (throw (IllegalArgumentException. (str "Unknown mode (" mode ")")))))

(defn run-tests
  "Runs multiple tests in a single mode. Returns a lazy sequence of test
  results."
  [mode & tests]
  (map (partial run-test mode) tests))

(defn export-for-mathematica
  "Given a log, exports information to files on disk about the
  parameter vectors and gradient, in a format compatible with
  the PathVisualizer.nb Mathematica applet. desc should be a
  human-readable description of the log, like \"ADADELTA with
  ε = 1e-3\". This will show up in the drop-down menu in the
  Mathematica applet. Unique filenames are used to avoid
  overwriting any existing files. Note that for the Mathematica
  applet to detect all the files, their numbers must be
  consecutive. This is the default for this function, but if
  you start deleting files manually, you have to make sure that
  you don't break the consecutivity requirement."
  [log desc]
  (let [log (util/vectorize log)
        [parameter-fname gradient-fname]
        (loop [index 1]
          (let [parameter-fname (str "resources/mathematica/opt" index "-param.txt")
                gradient-fname (str "resources/mathematica/opt" index "-grad.txt")]
            (if-not (and (.exists (io/as-file parameter-fname))
                         (.exists (io/as-file gradient-fname)))
              [parameter-fname gradient-fname]
              (recur (inc index)))))]
    (->> log
      (map :params)
      (map (partial str/join ","))
      (cons desc)
      (str/join \newline)
      (spit parameter-fname))
    (->> log
      (map :gradient)
      (map (partial str/join ","))
      (cons desc)
      (str/join \newline)
      (spit gradient-fname))))

;;;; Termination function builders

(defmacro term
  "Constructs a termination function from a collection of transformations.
  Each transformation is a higher-order function that takes a termination
  function (and optionally some additional parameters) and returns another
  termination function. To make it easy to reason about the order in
  which transformations are applied, transformations are grouped into
  two types: 'in' and 'out'. Conceptually, 'in' transformations modify the
  argument passed to the termination function before calling the original
  function; 'out' transformations call the original termination function,
  optionally cause side effects, and optionally modify the return value.

  The syntax for labels+transforms is as follows. Each element must be
  either a transformation function or one of the keywords :in or :out.
  Whether a transformation function is treated as an 'in' or an 'out'
  transformation is determined by looking at the last keyword specified
  in the argument vector. It is an error to specify a transformation as
  the first argument, since it cannot be determined which type the
  transformation is meant to be.

  The threading order of the transformations is designed to be as intuitive
  as possible. That is, provided that standard transformation functions are
  used, when the final termination function is called, first each of the
  'in' transformations are applied to its argument, from left to right,
  then each of the 'out' labels take effect, with the order of the side
  effects being from left to right and the rightmost modifications to the
  return value being last.

  This macro expands into a call to the single-arrow threading macro (->),
  so you can provide additional arguments to transformation functions by
  wrapping them in lists, just like with regular -> calls.

  If you create additional transformation functions, be sure to conform to
  the interface expected by this macro, which is as follows:

  - 'in' transformations should transform the argument, call the original
    termination function with the transformed argument, and return whatever
    is returned by original termination function.
  - 'out' transforms should call the original termination function with the
    given argument, and then terminate if the original termination function
    returns truthy. If the original termination function returns falsy,
    then they should perform any side effects, and return the possibly
    transformed or overridden return value.

  This will provide a consistent user experience at the REPL."
  [& labels+transforms]
  (when (and (seq labels+transforms)
             (not (keyword? (first labels+transforms))))
    (throw
      (IllegalArgumentException.
        (str "If you provide arguments, the first argument must be a keyword (was "
             (first labels+transforms)
             ")"))))
  (let [{:keys [in out]}
        (reduce (fn [xform-map [kws xforms]]
                  (doseq [kw kws]
                    (when-not (#{:in :out} kw)
                      (throw
                        (IllegalArgumentException.
                          (str "Invalid keyword, must be :in or :out (was "
                               kw
                               ")")))))
                  (update xform-map (last kws) (partial apply (fnil conj [])) xforms))
                {}
                (->> labels+transforms
                  (partition-by keyword?)
                  (partition 2)))]
    `(-> (constantly false) ~@out ~@(reverse in))))

;;;; Termination function 'in' transformations

(defn skip-initial
  "Skips the initial step for any transformations further down the
  pipe. This may be useful because certain pieces of information are
  not available on the first step (such as the last step). For
  instance, you may want to terminate when the step size falls below
  a certain magnitude, but this check will yield a NPE on the initial
  step unless you have skip-initial earlier in the transformation
  pipeline. You can also skip n steps, in general, by passing n."
  ([terminate?]
   (skip-initial terminate? 1))
  ([terminate? n]
   (let [num-calls* (atom 0)]
     (fn [state]
       (when (> (swap! num-calls* inc) n)
         (terminate? state))))))

(defn stagger
  "Transforms the argument into a map where :current is the original
  argument, and :last is the argument from the previous call (or nil
  if it is on the initial step)."
  [terminate?]
  (let [last-state* (atom nil)]
    (fn [state]
      (let [last-state @last-state*]
        (terminate?
          {:last last-state
           :current (reset! last-state* state)})))))

(defn count-steps
  "Adds a :step-count key to the argument, which is assumed to be
  a map, for any transformations further down the pipe. The step
  count is the number of steps that have happened previously, so
  on the initial step it will be 0."
  [terminate?]
  (let [num-steps* (atom -1)]
    (fn [state]
      (terminate? (assoc state :step-count (swap! num-steps* inc))))))

;;;; Termination function 'out' transformations -- return value modifications

(defn limit-steps
  "Stop after the given number of steps, or when the original
  termination function returns truthy."
  [terminate? n]
  (let [num-steps* (atom 0)]
    (fn [state]
      (swap! num-steps* inc)
      (or (terminate? state)
          (> @num-steps* n)))))

(defn limit-params
  "Stop after the distance between the parameter vector and the
  target vector is less than the given cutoff, or when the original
  termination function returns truthy."
  [terminate? cutoff target]
  (fn [state]
    (or (terminate? state)
        (<= (m/distance (- (:params state)
                           target))
            cutoff))))

(defn limit-value
  "Stop after the objective function value drops below the given
  cutoff, or when the original termination function returns truthy."
  [terminate? cutoff]
  (fn [state]
    (or (terminate? state)
        (<= (:value state)
            cutoff))))

(defn limit-gradient
  "Stop after the magnitude of the gradient drops below the given
  cutoff, or when the original termination function returns truthy."
  [terminate? cutoff]
  (fn [state]
    (or (terminate? state)
        (<= (m/magnitude (:gradient state))
            cutoff))))

;;;; Termination function 'out' transformations -- side effects

(defn slow-steps
  "Waits the specified number of milliseconds."
  [terminate? msecs]
  (fn [state]
    (when-not (terminate? state)
      (Thread/sleep msecs))))

(defn outputf
  "Prints information from the data structure passed to the termination
  function. Each of the args should be a function that can be called on
  the data structure in order to return the value that should be printed.
  Alternatively, args can be keys to look up in the data structure,
  which should then be associative. Or, args can be vectors, in which
  case the first function or key is used to get an object from the
  original data structure, then the second function or key is applied
  (or looked up) on the result, and so on, like get-in but allowing for
  functions in the vector. The results obtained by applying each arg to
  the data structure are passed to util/sformat along with fmt to
  determine the string to print. Note that you will have to provide a
  trailing %n if you want a newline to be printed at the end of the line.
  However, flushing is done automatically."
  [terminate? fmt & args]
  (fn [state]
    (or (terminate? state)
        (do
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
          false))))

(defn output
  "Same as outputf, but with the format as '%s %s ... %s'."
  [terminate? & args]
  (apply outputf terminate?
         (str (str/join \space
                        (repeat (count args) "%s"))
              "%n")
         args))

(defn stdout
  "Same as outputf, but with a predefined fmt and args that print the
  value of the objective function, the magnitude of the gradient, and
  the magnitude of the last step in an easy-to-read tabular format."
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
  "Waits for user input after each period (defaults to 1) steps. When
  prompted for input, press enter to run another period steps, enter
  a new positive integer to change the period, or enter something
  non-numeric to terminate."
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
