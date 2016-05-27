## Cortex Design

### Dataset Management

* loading, saving, serializing, streaming

* dataset protocols
 - PDatasetSamples: get the samples (core.matrix?)
 - PDatasetLabels: get the labels (core.matrix?)
 - PDatasetClasses: get a map of class ID -> string name
 - PDatasetMetadata: get metadata map that has description, source, etc...

* or just use a map?  {:samples, :labels, :classes, :metadata}

* utilities for generating datasets from text files
 - tf/idf

### Metrics and Hash Functions

* pull in metrics from bluejay
 - L1, L2, hamming, ...

* pull in LSH projections from bluejay
 - gaussian, bitwise, simhash, minhash, ...

### Machine Learning

* cross validation
 - provide helpers to separate a dataset into train, test, validate
 - k-fold cross validation helpers (also stratified, maintaining the same class
   balance as in the full set)
   * e.g. from sklearn.model_selection import StratifiedKFold

* grid search
 - exhaustive search across parameter space
 - random sampling search
 - smart sampling using Gaussian processes or some kind of importance sampling
 - gaussian process optimization ala spearmint?

* model serialization
 - extend model serialization protocols to other forms of models

* metrics: assess prediction errors for different model types
 - classification
 - multilabel ranking
 - regression
 - clustering

* training a model

(defprotocol PNeuralTraining
  "Protocol for modules that can be trained with forward / back propagation."
  (forward [this input]
    "Run a forward computation pass and return the updated module. output will be
    available for later retrieval. Input and intermediate states will be stored for
    futuere backward pass usage.")

  (backward [this input output-gradient]
    "Back propagate errors through the module with respect to the input.  Returns the
    module with input-gradient set (gradient at the inputs). Input must be the same
    as used in the forward pass.")

  (input-gradient [this]
    "Gets the computed input gradients for a module. Assumes the backward pass has been run."))

(defprotocol PTraining
  "Protocol for modules that can be trained input / output pairs."
  (train [this input output]
    "Trains the module to produce the given input / output. Accumulates gradients as necessary.
     Intended for use with update-parameters after completion of a (mini-)batch."))

* sklearn uses (fit <model> <samples> <targets>) and (estimate <model> <sample>)

### Supervised Learning

#### Regression
* linear regression (single linear layer)
* mlp regression
* regularized regression
* knn regression
 - using k-means
 - using LSH
* decision tree
* random forest

* implement multi-output meta-regression that trains many single output
regression models and then combines them into a meta-model
 - from sklearn.datasets import make_regression
  >>> from sklearn.multioutput import MultiOutputRegressor
  >>> from sklearn.ensemble import GradientBoostingRegressor
  >>> X, y = make_regression(n_samples=10, n_targets=3, random_state=1)
  >>> MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(X, y).predict(X)


#### Classification
* logistic regression (linear layer plus sigmoid, softmax?)
* mlp classification
* conv-net classification
* knn classifier
 - using k-means
 - using LSH
* decision tree classifier
* naive bayes classifier
* random forest classifier
* SVM using a margin type loss on a neural network

* multi-class meta-classifier
 - train many underlying binary classifiers

### Unsupervised Learning

#### Clustering
* k-means
* k-means++
* LSH bins
* LSH accelerated k-means?
* agglomerative
* LSH accelerated neighbor based clustering
* mini-batch meta-clusterer
 - sample subsets of the data for working with really large datasets

#### Dimensionality Reduction

* linear autoencoder (~ PCA)
* PCA
* ICA
* non-linear autoencoder
* sparse autoencoder
* denoising autoencoder

#### Density Estimation

* histogram

#### Vector Quantization

* kohonen map
* neural gas

#### Onyx Based Distributed Versions

* could be a commercial product that we sell
* implements same interfaces as the built-in algorithms
* scales insanely, real-time, etc...
* let users create an account, then pass in credentials to the built-in
  functions which will then fit the model and run in the cloud on a giant
  cluster spun up for them.

