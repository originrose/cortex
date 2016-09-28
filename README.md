# Cortex ![TravisCI](https://travis-ci.com/thinktopic/cortex.svg?token=pNFS4aJt3yqGNNwZvG5z&branch=master)

Neural networks, regression and feature learning in Clojure.

### TODO:

* more activation functions (e.g. tanh)
* more loss functions (e.g. negative log likelihood)
* more layer types (e.g. dropout)

* composite layer types
 - port additive and multiplicative layers CMulTable, CAddTable from Torch for recurrent nets and LSTM
 
### See also:

`ROADMAP.md`

## Gradient descent

`cortex` contains a sub-library for performing instrumented gradient descent, which is located in the `cortex.optimise.*` namespaces. See the namespace docstring for `cortex.optimise.descent` for example usage.
