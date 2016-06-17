## Cortex Design

### Dataset Management

* loading, saving, serializing, streaming

* dataset protocols
 - PDatasetSamples: get the samples (core.matrix?)
 - PDatasetLabels: get the labels (core.matrix?)
 - PDatasetClasses: get a map of class ID -> string name
 - PDatasetMetadata: get metadata map that has description, source, etc...

* or just use a map?  {:samples, :labels, :classes, :metadata}
  * or map with schema/spec?

* utilities for generating datasets from text files
 - tf/idf
 - GloVe - looks like it's in [dl4j](https://github.com/deeplearning4j/dl4j-0.4-examples/blob/master/src/main/java/org/deeplearning4j/examples/nlp/glove/GloVeExample.java)

### Preprocessing

* preprocessing: scaling and encoding, example set of routines [here](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing)

### Metrics and Hash Functions

* pull in metrics from bluejay
 - L1, L2, hamming, ...

* pull in LSH projections from bluejay
 - gaussian, bitwise, simhash, minhash, ...

### Machine Learning

* A useful top level enumeration of functionality for reference is the sklearn [top level API doc](http://scikit-learn.org/stable/modules/classes.html).

* cross validation
 * provide helpers to separate a dataset into train, test, validate
 * k-fold cross validation helpers (also stratified, maintaining the same class
   balance as in the full set)
    * e.g. from sklearn.model_selection import StratifiedKFold
 * `sklearn` can be verbose and painful here, would be nice to destructure
   by position or keys as in a doseq or let for a particular scope defined
   by the kfold (macro?)
 * Manual indexing in particular is painful, a destructuring in a `let` style way to:
   * indicate that you're in the scope of a k-fold cv that will iterate over ks
   * indicate in-fold and out-of-fold partitions of the data
   * be able to make out-of-fold predictions (e.g. for ensembles where you need to
     avoid information leakage)

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
  * further breakdown:
    * all models `fit`
    * PCA, etc. `transform`
    * preprocessors (e.g. linear scaling) also use `transform`
    * some regressors `estimate`
    * classifiers implement `predict` and `predict_proba` (goofy name, class probability)
    * clustering a la K Means, GMM use `predict` as well.

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

#### Ensembles

* AdaBoost
* hooks for [Java](https://github.com/dmlc/xgboost/tree/master/jvm-packages)[XGBoost](https://github.com/dmlc/xgboost)?
* implement multi-output meta-regression that trains many single output
regression models and then combines them into a meta-model
* MultiOutputRegressor a la [sklearn](http://scikit-learn.org/dev/modules/generated/sklearn.multioutput.MultiOutputRegressor.html):

```python
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
X, y = make_regression(n_samples=10, n_targets=3, random_state=1)
MultiOutputRegressor(GradientBoostingRegressor(random_state=0)).fit(X, y).predict(X)
```
* A meta-ensembler that supports boosting/bagging/blending methods as described [here](http://mlwave.com/kaggle-ensembling-guide/)
  * Something like this could be a component of a dataflow graph whose execution model could support
    caching. 


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
* GMM
* DBSCAN

#### Dimensionality Reduction

* linear autoencoder (~ PCA)
* PCA
* ICA
* non-linear autoencoder
* sparse autoencoder
* denoising autoencoder
* t-SNE

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

### High Level Model and Hyperparameter Abstractions

You want parameter search methods to have access to:

* Model generation (using hyperparameters), e.g. HOF/Ctor for Model, parameters of inputs
* Data flow description (preprocessing first, etc.?)
* Accuracy/Loss (model level differs from NN optimizers, useful for scoring models in
  ensembles a la Boosting model weights as well as parameter search.)

These abstractions could be protocol/interface level, data descriptors a la `spec`, etc. Ideally
you could pipe any of this data flow description into a DAG:

* An execution model that can cache or otherwise take advantage of previously executed work.
  * Kagglers manually hack this in by ensembling CSVs, [example](https://github.com/MLWave/Kaggle-Ensemble-Guide/blob/master/kaggle_avg.py)
* An execution model that can evaluate model components for search in parallel as appropriate.

### Existing Framework Comparisons

* Stanford CS 231 [Lecture 12](http://cs231n.stanford.edu/slides/winter1516_lecture12.pdf) contains a detailed
  breakdown of Caffe, Torch, Theano, and TensorFlow.
