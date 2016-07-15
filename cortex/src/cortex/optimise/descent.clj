(ns cortex.optimise.descent
  "Contains API functions for performing gradient descent on pure
  functions using gradient optimisers."
  (:require [cortex.nn.protocols :as cp]
            [cortex.optimise.functions]
            [cortex.optimise.optimisers]))

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
