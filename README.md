# Cortex

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

`cortex` contains a gradient descent sub-library, which is located in the `cortex.optimise.*` namespaces. Its interface is currently undergoing rapid change, but more documentation will be coming soon.