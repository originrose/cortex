## Cortex Next Steps

### Preprocessing

* preprocessing: scaling and encoding, example set of routines [here](http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing)

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


* sklearn uses (fit <model> <samples> <targets>) and (estimate <model> <sample>)
  * further breakdown:
    * all models `fit`
    * PCA, etc. `transform`
    * preprocessors (e.g. linear scaling) also use `transform`
    * some regressors `estimate`
    * classifiers implement `predict` and `predict_proba` (goofy name, class probability)
    * clustering a la K Means, GMM use `predict` as well.


#### Dimensionality Reduction

* linear autoencoder (~ PCA)
* PCA
* ICA
* non-linear autoencoder
* sparse autoencoder
* denoising autoencoder
* t-SNE

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