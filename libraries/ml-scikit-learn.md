## [Machine Learning] | [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#) | [github](https://github.com/scikit-learn/scikit-learn)

- [Supervised Learning: Classification](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#classification)
- [Supervised Learning: Regression](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#regression)
- [Unsupervised Learning](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#unsupervised-learning)
  - [cluster](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#unsupervised-learning-cluster)
  - [dimensionality-reduction](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#unsupervised-learning-dimensionality-reduction)
  - [decomposing-signals-in-components](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#unsupervised-learning-decomposing-signals-in-components)
  - [gaussian-mixture](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#unsupervised-learning-gaussian-mixture)
  - [density-estimation](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#unsupervised-learning-density-estimation)
  - [covariance-estimation](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#unsupervised-learning-covariance-estimation)
- [Feature Selection](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#feature-selection)
- [Model Selection](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#model-selection)
  - [dataset-spliter](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#model-selection-dataset-spliter)
  - [hyper-parameter-optimizers](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#model-selection-hyper-parameter-optimizers)
- [Preprocessing](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#preprocessing)
  - [preprocessing-transformer](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#preprocessing-transformer)
  - [preprocessing-scaler](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#preprocessing-scaler)
- [Datasets](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#datasets)
  - [datasets: real-world](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#datasets-real-world)
  - [datasets: simulation](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#datasets-samples-generator)
- [Metrics](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#metrics)
  - [metrics-classification](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#metrics-classification)
  - [metrics-regression](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#metrics-regression)
  - [metrics-clustering](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#metrics-clustering)
- [Pipeline](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#Pipeline)

---

### Classification
#### Classification: linear_model
- https://scikit-learn.org/stable/modules/linear_model.html
- https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model

`[Classification]: LogisticRegression`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import linear_model

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = linear_model.LogisticRegression(penalty='l2', max_iter=500)
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```

#### Classification: ensemble
- https://scikit-learn.org/stable/modules/ensemble.html
- https://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble

`[Classfication]: AdaBoostClassifier`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import ensemble

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = ensemble.AdaBoostClassifier()
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
`[Classification]: GradientBoostingClassifier`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import ensemble

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = ensemble.GradientBoostingClassifier()
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
`[Classfication]: BaggingClassifier`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import ensemble

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = ensemble.BaggingClassifier()
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
`[Classfication]: ExtraTreesClassifier`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import ensemble

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = ensemble.ExtraTreesClassifier()
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
`[Classfication]: RandomForestClassifier`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import ensemble

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = ensemble.RandomForestClassifier()
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
`[Classfication]: VotingClassifier`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import tree, ensemble

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
estimators = [('Base1', tree.DecisionTreeClassifier()),
              ('Base2', tree.DecisionTreeClassifier()),
              ('Base3', tree.DecisionTreeClassifier())]
classifier = ensemble.VotingClassifier(estimators=estimators)
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
`[Classfication]: StackingClassifier`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import tree, ensemble

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
estimators = [('Base1', tree.DecisionTreeClassifier()),
              ('Base2', tree.DecisionTreeClassifier()),
              ('Base3', tree.DecisionTreeClassifier())]
meta_classifier = tree.DecisionTreeClassifier()
classifier = ensemble.StackingClassifier(estimators=estimators, final_estimator=meta_classifier)
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```

#### Classification: naive_bayes
- https://scikit-learn.org/stable/modules/naive_bayes.html
- https://scikit-learn.org/stable/modules/classes.html#module-sklearn.naive_bayes

`[Classification]: BernoulliNB`
```python
#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html#sklearn.naive_bayes.BernoulliNB
import joblib
from ailever.dataset import SKAPI
from sklearn import naive_bayes

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = naive_bayes.BernoulliNB()
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
`[Classification]: MultinomialNB`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import naive_bayes

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = naive_bayes.MultinomialNB()
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
`[Classification]: GaussianNB`
```python
#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
import joblib
from ailever.dataset import SKAPI
from sklearn import naive_bayes

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = naive_bayes.GaussianNB()
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
`[Classification]: CategoricalNB`
```python
#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.CategoricalNB.html#sklearn.naive_bayes.CategoricalNB
import joblib
from ailever.dataset import SKAPI
from sklearn import naive_bayes

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = naive_bayes.CategoricalNB()
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
`[Classification]: ComplementNB`
```python
#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html#sklearn.naive_bayes.ComplementNB
import joblib
from ailever.dataset import SKAPI
from sklearn import naive_bayes

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = naive_bayes.ComplementNB()
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```


#### Classification: tree
- https://scikit-learn.org/stable/modules/tree.html
- https://scikit-learn.org/stable/modules/classes.html#module-sklearn.tree

`[Classification]: DecisionTreeClassifier`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import tree

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = tree.DecisionTreeClassifier()
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
`[Classification]: ExtraTreeClassifier`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import tree

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = tree.ExtraTreeClassifier()
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```

#### Classification: neighbors
- https://scikit-learn.org/stable/modules/neighbors.html
- https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors

`[Classification]: KNeighborsClassifier`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import neighbors

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = neighbors.KNeighborsClassifier()
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
`[Classification]: RadiusNeighborsClassifier`
```
import joblib
from ailever.dataset import SKAPI
from sklearn import neighbors

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = neighbors.RadiusNeighborsClassifier()
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```

#### Classification: discriminant_analysis
- https://scikit-learn.org/stable/modules/lda_qda.html
- https://scikit-learn.org/stable/modules/classes.html#module-sklearn.discriminant_analysis

`[Classification]: LinearDiscriminantAnalysis`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import discriminant_analysis

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = discriminant_analysis.LinearDiscriminantAnalysis()
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
`[Classification]: QuadraticDiscriminantAnalysis`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import discriminant_analysis

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = discriminant_analysis.QuadraticDiscriminantAnalysis()
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
#### Classification: svm
- https://scikit-learn.org/stable/modules/svm.html
- https://scikit-learn.org/stable/modules/classes.html#module-sklearn.svm

`[Classification]: LinearSVC`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import svm

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = svm.LinearSVC(max_iter=10000)
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
`[Classification]: SVC`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import svm

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = svm.SVC(kernel='poly', max_iter=10000) # kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
`[Classification]: NuSVC`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import svm

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = svm.NuSVC(max_iter=1000)
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```

#### Classification: neural_network
- https://scikit-learn.org/stable/modules/neural_networks_supervised.html
- https://scikit-learn.org/stable/modules/classes.html#module-sklearn.neural_network

`[Classification]: MLPClassifier`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import neural_network

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = neural_network.MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, activation='relu', solver='adam', learning_rate='adaptive')
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
```
`[Classification]: `
```python
```
`[Classification]: `
```python
```
`[Classification]: `
```python
```
`[Classification]: `
```python
```
`[Classification]: `
```python
```
`[Classification]: `
```python
```


### Regression
#### Regression: linear_model
`[Regression]: LinearRegression`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import linear_model

# [STEP1]: data
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target']
y = dataset.loc[:, dataset.columns == 'target']

# [STEP2]: model
regressor = linear_model.LinearRegression()
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
#regressor.coef_
#regressor.intercept_
regressor.predict(X[0:10])
```
`[Regression]: Ridge`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import linear_model

# [STEP1]: data
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target']
y = dataset.loc[:, dataset.columns == 'target']

# [STEP2]: model
regressor = linear_model.Ridge(alpha=.5)
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
#regressor.coef_
#regressor.intercept_
regressor.predict(X[0:10])
```
`[Regression]: Lasso`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import linear_model

# [STEP1]: data
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target']
y = dataset.loc[:, dataset.columns == 'target']

# [STEP2]: model
regressor = linear_model.Lasso(alpha=0.1)
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
#regressor.coef_
#regressor.intercept_
regressor.predict(X[0:10])
```
`[Regression]: LassoLars`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import linear_model

# [STEP1]: data
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target']
y = dataset.loc[:, dataset.columns == 'target']

# [STEP2]: model
regressor = linear_model.LassoLars(alpha=.1, normalize=False)
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
#regressor.coef_
#regressor.intercept_
regressor.predict(X[0:10])
```
`[Regression]: ElasticNet`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import linear_model

# [STEP1]: data
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target']
y = dataset.loc[:, dataset.columns == 'target']

# [STEP2]: model
regressor = linear_model.ElasticNet()
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
#regressor.coef_
#regressor.intercept_
regressor.predict(X[0:10])
```
`[Regression]: BayesianRidge`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import linear_model

# [STEP1]: data
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
regressor = linear_model.BayesianRidge()
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
#regressor.coef_
#regressor.intercept_
regressor.predict(X[0:10])
```

#### Regression: ensemble
`[Regression]: AdaBoostRegressor`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import ensemble

# [STEP1]: data
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
regressor = ensemble.AdaBoostRegressor()
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```
`[Regression]: GradientBoostingRegressor`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import ensemble

# [STEP1]: data
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
regressor = ensemble.GradientBoostingRegressor()
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```
`[Regression]: BaggingRegressor`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import ensemble

# [STEP1]: data
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
regressor = ensemble.BaggingRegressor()
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```
`[Regression]: ExtraTreesRegressor`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import ensemble

# [STEP1]: data
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
regressor = ensemble.ExtraTreesRegressor()
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```
`[Regression]: RandomForestRegressor`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import ensemble

# [STEP1]: data
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
regressor = ensemble.RandomForestRegressor()
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```
`[Regression]: VotingRegressor`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import ensemble

# [STEP1]: data
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
estimators = [('Base1', tree.DecisionTreeRegressor()),
              ('Base2', tree.DecisionTreeRegressor()),
              ('Base3', tree.DecisionTreeRegressor())]
regressor = ensemble.VotingRegressor(estimators=estimators)
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```
`[Regression]: StackingRegressor`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import ensemble

# [STEP1]: data
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
estimators = [('Base1', tree.DecisionTreeRegressor()),
              ('Base2', tree.DecisionTreeRegressor()),
              ('Base3', tree.DecisionTreeRegressor())]
regressor = ensemble.StackingRegressor(estimators=estimators)
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```

#### Regression: neighbors
`[Regression]: KNeighborsRegressor`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import neighbors

# [STEP1]: data
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target']
y = dataset.loc[:, dataset.columns == 'target']

# [STEP2]: model
regressor = neighbors.KNeighborsRegressor()
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```
`[Regression]: RadiusNeighborsRegressor`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import neighbors

# [STEP1]: data
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target']
y = dataset.loc[:, dataset.columns == 'target']

# [STEP2]: model
regressor = neighbors.RadiusNeighborsRegressor()
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```

#### Regression: tree
`[Regression]: DecisionTreeRegressor`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import tree

# [STEP1]: data
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
regressor = tree.DecisionTreeRegressor()
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```
`[Regression]: ExtraTreeRegressor`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import tree

# [STEP1]: data
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
regressor = tree.ExtraTreeRegressor()
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```

#### Regression: svm
`[Regression]: LinearSVR`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import svm

# [STEP1]: data
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
regressor = svm.LinearSVR(max_iter=1000)
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```
`[Regression]: SVR`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import svm

# [STEP1]: data
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
regressor = svm.SVR(kernel='poly', max_iter=1000) # kernel: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’}, default=’rbf’
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```
`[Regression]: NuSVR`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import svm

# [STEP1]: data
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
regressor = svm.NuSVR(max_iter=1000)
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```
#### Regression: neural_network
`[Regression]: MLPRegressor`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import neural_network

# [STEP1]: data
dataset = SKAPI.housing(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
regressor = neural_network.MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, activation='relu', solver='adam', learning_rate='adaptive')
regressor.fit(X, y)

# [STEP3]: save & load
joblib.dump(regressor, 'regressor.joblib')
regressor = joblib.load('regressor.joblib')

# [STEP4]: prediction
regressor.predict(X[0:10])
```
`[Regression]: `
```python
```
`[Regression]: `
```python
```



### Unsupervised Learning
#### Unsupervised Learning: cluster
`[Unsupervised Learning]: kmeans clustering`  
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(20)
X = np.random.normal(size=(1000, 2)) # 1000 pts in 2

# training
model = KMeans(n_clusters=3)
model.fit(X)
y_pred = model.predict(X) # labels for each of the points
centers = model.cluster_centers_ # locations of the clusters
labels = model.labels_

# visualization
center1 = np.where(y_pred == 0)
center2 = np.where(y_pred == 1)
center3 = np.where(y_pred == 2)

fig, axes = plt.subplots(1,2, figsize=(15,10))
axes[0].scatter(X[:,0], X[:,1])
axes[0].grid()
axes[1].scatter(X[:,0][center1], X[:,1][center1])
axes[1].scatter(X[:,0][center2], X[:,1][center2])
axes[1].scatter(X[:,0][center3], X[:,1][center3])
axes[1].grid()
plt.show()
```

#### Unsupervised Learning: Dimensionality Reduction
`[Dimensionality Reduction]: Isomap`
```python
import numpy as np
from sklearn.manifold import Isomap

X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

model = Isomap(n_components=2)
X_embeded = model.fit_transform(X); print(X_embeded)
```
`[Dimensionality Reduction]: LocallyLinearEmbedding-hessian`
```python
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding

X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

model = LocallyLinearEmbedding(n_neighbors=6, n_components=2, method='hessian')
X_embeded = model.fit_transform(X[:100]); print(X_embeded)
```
`[Dimensionality Reduction]: LocallyLinearEmbedding-ltsa`
```python
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding

X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

model = LocallyLinearEmbedding(n_neighbors=5, n_components=2, method='ltsa')
X_embeded = model.fit_transform(X[:100]); print(X_embeded)
```
`[Dimensionality Reduction]: LocallyLinearEmbedding-modified`
```python
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding

X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

model = LocallyLinearEmbedding(n_neighbors=5, n_components=2, method='modified')
X_embeded = model.fit_transform(X[:100]); print(X_embeded)
```
`[Dimensionality Reduction]: LocallyLinearEmbedding-standard`
```python
import numpy as np
from sklearn.manifold import LocallyLinearEmbedding

X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

model = LocallyLinearEmbedding(n_neighbors=5, n_components=2, method='standard')
X_embeded = model.fit_transform(X[:100]); print(X_embeded)
```
`[Dimensionality Reduction]: MDS`
```python
import numpy as np
from sklearn.manifold import MDS

X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

model = MDS(n_components=2)
X_embeded = model.fit_transform(X[:100]); print(X_embeded)
```
`[Dimensionality Reduction]: SpectralEmbedding`
```python
import numpy as np
from sklearn.manifold import SpectralEmbedding

X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

model = SpectralEmbedding(n_components=2)
X_embeded = model.fit_transform(X); print(X_embeded)
```
`[Dimensionality Reduction]: TSNE`
```python
import numpy as np
from sklearn.manifold import TSNE

X = np.array([[0, 0, 0],
              [0, 0, 1],
              [0, 1, 0],
              [0, 1, 1],
              [1, 0, 0],
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

model = TSNE(n_components=2)
X_embeded = model.fit_transform(X); print(X_embeded)
```

#### Unsupervised Learning: Decomposing signals in components
`[Decomposing signals in components]: PCA`
```python
from ailever.dataset import SMAPI
from sklearn.decomposition import PCA

dataset = SMAPI.macrodata(download=False)
X = dataset.loc[:, dataset.columns != 'infl'].values
#y = dataset.loc[:, dataset.columns == 'infl'].values.ravel()

decomposition = PCA(n_components=2, svd_solver='full') # svd_solver: {‘auto’, ‘full’, ‘arpack’, ‘randomized’}, default='auto'
decomposition.fit(X) # #print(decomposition.components_) #: importance[dim0:raw_feature, dim1:selected_feature] 
print(decomposition.explained_variance_ratio_)
print(decomposition.singular_values_)

X_new = decomposition.fit_transform(X)
print(X.shape, X_new.shape)
```
`[Decomposing signals in components]: KernelPCA`
```python
from ailever.dataset import SMAPI
from sklearn.decomposition import KernelPCA

dataset = SMAPI.macrodata(download=False)
X = dataset.loc[:, dataset.columns != 'infl'].values
#y = dataset.loc[:, dataset.columns == 'infl'].values.ravel()

decomposition = KernelPCA(n_components=7, kernel='linear', fit_inverse_transform=True) # kernel: {‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘cosine’, ‘precomputed’}, default=’linear’
decomposition.fit(X) 

X_new = decomposition.fit_transform(X)
X_ = decomposition.inverse_transform(X_new)
print(X.shape, X_new.shape, X_.shape)
```
`[Decomposing signals in components]: FastICA`
```python
from ailever.dataset import SMAPI
from sklearn.decomposition import FastICA

dataset = SMAPI.macrodata(download=False)
X = dataset.loc[:, dataset.columns != 'infl'].values
#y = dataset.loc[:, dataset.columns == 'infl'].values.ravel()

decomposition = FastICA(n_components=4, random_state=0)
decomposition.fit(X) #print(decomposition.components_) #: importance[dim0:raw_feature, dim1:selected_feature]

X_new = decomposition.fit_transform(X)
X_ = decomposition.inverse_transform(X_new)
print(X.shape, X_new.shape, X_.shape)
```
`[Decomposing signals in components]: TruncatedSVD`
```python
from ailever.dataset import SMAPI
from sklearn.decomposition import TruncatedSVD

dataset = SMAPI.macrodata(download=False)
X = dataset.loc[:, dataset.columns != 'infl'].values
#y = dataset.loc[:, dataset.columns == 'infl'].values.ravel()

decomposition = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
decomposition.fit(X) #print(decomposition.components_) #: importance[dim0:raw_feature, dim1:selected_feature]
print(decomposition.explained_variance_ratio_)
print(decomposition.singular_values_)

X_new = decomposition.fit_transform(X)
print(X.shape, X_new.shape)
```
`[Decomposing signals in components]: FactorAnalysis`
```python
from ailever.dataset import SMAPI
from sklearn.decomposition import FactorAnalysis

dataset = SMAPI.macrodata(download=False)
X = dataset.loc[:, dataset.columns != 'infl'].values
#y = dataset.loc[:, dataset.columns == 'infl'].values.ravel()

decomposition = FactorAnalysis(n_components=7, random_state=0, rotation="varimax") # rotation=None or "varimax"
decomposition.fit(X) #print(decomposition.components_) #: importance[dim0:raw_feature, dim1:selected_feature]

X_new = decomposition.fit_transform(X)
print(X.shape, X_new.shape)
```

#### Unsupervised Learning: Gaussian Mixture
`[Gaussian Mixture]: `
```python
```

#### Unsupervised Learning: Density Estimation
`[Density Estimation]: `
```python
```

#### Unsupervised Learning: Covariance Estimation
`[Covariance estimation]: `
```python
```


### Feature Selection
`[Feature Selection]: VarianceThreshold`
```python
from sklearn.feature_selection import VarianceThreshold

X = np.array([[0, 0, 1], 
              [0, 1, 0], 
              [1, 0, 0], 
              [0, 1, 1], 
              [0, 1, 0], 
              [0, 1, 1]])

# Threshold: Var[X] = p*(1-p)
selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
X_new = selector.fit_transform(X)
print(X.shape, X_new.shape)
```
`[Feature Selection]: r(coefficient of determination), f-test(anova, regression), mutual-information`
```python
import numpy as np
from ailever.dataset import SKAPI
from sklearn.feature_selection import f_classif, r_regression, f_regression, mutual_info_classif

dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

f_test_by_anova, _ = f_classif(X,y)
f_test_by_anova /= np.max(f_test_by_anova)
print(f_test_by_anova)

r = r_regression(X, y)
print(r) # Pearson’s r

f_test_by_regression, _ = f_regression(X, y)
f_test_by_regression /= np.max(f_test_by_regression)
print(f_test_by_regression)

mi = mutual_info_classif(X, y)
mi /= np.max(mi)
print(mi)
```
`[Feature Selection]: chi2`
```python
from ailever.dataset import SKAPI
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

selector = SelectKBest(chi2, k=2)
X_new = selector.fit_transform(X, y)
print(X.shape, X_new.shape)
```
`[Feature Selection]: RFE`
```python
from ailever.dataset import SKAPI
from sklearn import svm
from sklearn.feature_selection import RFE

dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

classifier = svm.SVC(kernel="linear", C=1)
selector = RFE(estimator=classifier, n_features_to_select=1, step=1)
selector.fit(X, y)
X_new = X[:, selector.ranking_ == 1]
print(X.shape, X_new.shape)
```
`[Feature Selection]: SelectFromModel(1) Sparse Estimator`
```python
"""
Linear models penalized with the L1 norm have sparse solutions: many of their estimated coefficients are zero. When the goal is to reduce the dimensionality of the data to use with another classifier, they can be used along with SelectFromModel to select the non-zero coefficients. In particular, sparse estimators useful for this purpose are the Lasso for regression, and of LogisticRegression and LinearSVC for classification:
"""
from ailever.dataset import SKAPI
from sklearn import svm
from sklearn.feature_selection import SelectFromModel

dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

classifier = svm.LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y) # Linear models penalized with the L1 norm: Lasso, LogisticRegression, LinearSVC
selector = SelectFromModel(classifier, prefit=True)
X_new = selector.transform(X)
print(X.shape, X_new.shape)
```
`[Feature Selection]: SelectFromModel(2) Tree-based Model(impurity-based feature importances)`
```python
from ailever.dataset import SKAPI
from sklearn import ensemble
from sklearn.feature_selection import SequentialFeatureSelector

dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

classifier = ensemble.ExtraTreesClassifier(n_estimators=50).fit(X, y) # impurity-based feature importances: classifier.feature_importances_
selector = SelectFromModel(classifier, prefit=True)
X_new = selector.transform(X)
print(X.shape, X_new.shape)
```
`[Feature Selection]: SequentialFeatureSelector(1) forward`
```python
from ailever.dataset import SKAPI
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SequentialFeatureSelector

dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

classifier = LassoCV().fit(X, y)
selector = SequentialFeatureSelector(classifier, n_features_to_select=2, direction="forward").fit(X,y)

X_new = X[:, selector.get_support()]
print(X.shape, X_new.shape,)
```
`[Feature Selection]: SequentialFeatureSelector(1) backward`
```python
from ailever.dataset import SKAPI
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SequentialFeatureSelector

dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

classifier = LassoCV().fit(X, y)
selector = SequentialFeatureSelector(classifier, n_features_to_select=2, direction="backward").fit(X,y)

X_new = X[:, selector.get_support()]
print(X.shape, X_new.shape,)
```


### Model Selection
- https://scikit-learn.org/stable/model_selection.html
- https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
- https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

#### Model Selection: Dataset-Spliter
**Classification Metrics**:
'accuracy', 'balanced_accuracy', 'top_k_accuracy', 'average_precision', 'neg_brier_score', 'f1', 'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples', 'neg_log_loss',  'precision', 'recall', 'jaccard', 'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted'

**Regression Metrics**:
'explained_variance', 'max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'r2', 'neg_mean_poisson_deviance', 'neg_mean_gamma_deviance', 'neg_mean_absolute_percentage_error'

**Clustering Metrics**:
'adjusted_mutual_info_score', 'adjusted_rand_score', 'completeness_score', 'fowlkes_mallows_score', 'homogeneity_score', 'mutual_info_score', 'normalized_mutual_info_score', 'rand_score', 'v_measure_score'


`[Model Selection]: LeaveOneOut`
```python
from sklearn.model_selection import LeaveOneOut

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
cross_validation = LeaveOneOut()
for train_index, test_index in cross_validation.split(X):
     print("%s %s" % (train_index, test_index))
```
```python
import matplotlib.pyplot as plt
from ailever.dataset import SKAPI
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# [STEP1]: data
cross_validation = LeaveOneOut()
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model evaluation
classifier = DecisionTreeClassifier()
cv_results = cross_val_score(classifier, X, y, cv=cross_validation, scoring='accuracy')
print('- score(mean):', cv_results.mean())
print('- score(std):', cv_results.std())

# [STEP3]: visualization
fig = plt.figure()
plt.boxplot(cv_results)
plt.show()
```

`[Model Selection]: KFold`
```python
from sklearn.model_selection import KFold

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
cross_validation = KFold(n_splits=3, shuffle=True)
for train_index, test_index in cross_validation.split(X):
     print("%s %s" % (train_index, test_index))
```
```python
import matplotlib.pyplot as plt
from ailever.dataset import SKAPI
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# [STEP1]: data
cross_validation = KFold(n_splits=10, shuffle=True, random_state=None)
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model evaluation
classifier = DecisionTreeClassifier()
cv_results = cross_val_score(classifier, X, y, cv=cross_validation, scoring='accuracy')
print('- score(mean):', cv_results.mean())
print('- score(std):', cv_results.std())

# [STEP3]: visualization
fig = plt.figure()
plt.boxplot(cv_results)
plt.show()
```

`[Model Selection]: StratifiedKFold`
```python
from sklearn.model_selection import StratifiedKFold

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
cross_validation = StratifiedKFold(n_splits=4, shuffle=True)
for train_index, test_index in cross_validation.split(X, y):
    print("%s %s" % (train_index, test_index))
```
```python
import matplotlib.pyplot as plt
from ailever.dataset import SKAPI
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# [STEP1]: data
cross_validation = StratifiedKFold(n_splits=4, shuffle=True)
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model evaluation
classifier = DecisionTreeClassifier()
cv_results = cross_val_score(classifier, X, y, cv=cross_validation, scoring='accuracy')
print('- score(mean):', cv_results.mean())
print('- score(std):', cv_results.std())

# [STEP3]: visualization
fig = plt.figure()
plt.boxplot(cv_results)
plt.show()
```

`[Model Selection]: GroupKFold`
```python
from sklearn.model_selection import GroupKFold

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
groups = ['a','a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'd', 'd', 'd', 'd', 'd', 'd']
gkf = GroupKFold(n_splits=4)
for train_index, test_index in gkf.split(X, y, groups=groups):
     print("%s %s" % (train_index, test_index))
```

`[Model Selection]: TimeSeriesSplit`
```python
from sklearn.model_selection import TimeSeriesSplit

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
cross_validation = TimeSeriesSplit()
for train_index, test_index in cross_validation.split(X):
     print("%s %s" % (train_index, test_index))
```
```python
import matplotlib.pyplot as plt
from ailever.dataset import SKAPI
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

# [STEP1]: data
cross_validation = TimeSeriesSplit()
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model evaluation
classifier = DecisionTreeClassifier()
cv_results = cross_val_score(classifier, X, y, cv=cross_validation, scoring='accuracy')
print('- score(mean):', cv_results.mean())
print('- score(std):', cv_results.std())

# [STEP3]: visualization
fig = plt.figure()
plt.boxplot(cv_results)
plt.show()
```


#### Model Selection: Hyper-parameter optimizers
`[Model Selection]: RandomizedSearchCV`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import ensemble
from sklearn.model_selection import RandomizedSearchCV

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
parameters={'n_estimators':[50, 100, 200],
            'learning_rate': [1, 0.1, 0.01],
            'algorithm': ['SAMME', 'SAMME.R']}
classifier = ensemble.AdaBoostClassifier()
classifier = RandomizedSearchCV(estimator=classifier, 
                                param_distributions=parameters, 
                                n_iter=10)
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
means = classifier.cv_results_['mean_test_score']
stds = classifier.cv_results_['std_test_score']
params = classifier.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#classifier.best_params_
#classifier.best_estimator_
#classifier.best_score_
```
`[Model Selection]: RandomizedSearchCV + CrossValidation`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import ensemble
from sklearn.model_selection import KFold
from sklearn.model_selection import RandomizedSearchCV

# [STEP1]: data
cross_validation = KFold(n_splits=10, shuffle=True, random_state=None)
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
parameters={'n_estimators':[50, 100, 200],
            'learning_rate': [1, 0.1, 0.01],
            'algorithm': ['SAMME', 'SAMME.R']}
classifier = ensemble.AdaBoostClassifier()
classifier = RandomizedSearchCV(estimator=classifier, 
                                param_distributions=parameters,
                                cv=cross_validation, 
                                n_iter=10)
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])
means = classifier.cv_results_['mean_test_score']
stds = classifier.cv_results_['std_test_score']
params = classifier.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#classifier.best_params_
#classifier.best_estimator_
#classifier.best_score_
```

`[Model Selection]: GridSearchCV`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import ensemble
from sklearn.model_selection import GridSearchCV

# [STEP1]: data
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
parameters={'n_estimators':[50, 100, 200],
            'learning_rate': [1, 0.1, 0.01],
            'algorithm': ['SAMME', 'SAMME.R']}
classifier = ensemble.AdaBoostClassifier()
classifier = GridSearchCV(estimator=classifier, 
                          param_grid=parameters)
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])

# [STEP5]: evaluation
print(classifier.cv_results_.keys())
means = classifier.cv_results_['mean_test_score']
stds = classifier.cv_results_['std_test_score']
params = classifier.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#classifier.best_params_
#classifier.best_estimator_
#classifier.best_score_
```

`[Model Selection]: GridSearchCV + CrossValidation`
```python
import joblib
from ailever.dataset import SKAPI
from sklearn import ensemble
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# [STEP1]: data
cross_validation = KFold(n_splits=10, shuffle=True, random_state=None)
dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
parameters={'n_estimators':[50, 100, 200],
            'learning_rate': [1, 0.1, 0.01],
            'algorithm': ['SAMME', 'SAMME.R']}
classifier = ensemble.AdaBoostClassifier()
classifier = GridSearchCV(estimator=classifier, 
                          param_grid=parameters,
                          cv=cross_validation)
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
classifier.predict(X[0:10])

# [STEP5]: evaluation
print(classifier.cv_results_.keys())
means = classifier.cv_results_['mean_test_score']
stds = classifier.cv_results_['std_test_score']
params = classifier.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

#classifier.best_params_
#classifier.best_estimator_
#classifier.best_score_
```


### Preprocessing
#### Preprocessing: Transformer

#### Preprocessing: Scaler
`[Preprocessing]: MaxAbsScaler`
```python
from sklearn.preprocessing import MaxAbsScaler

X = [[ 1., -1.,  2.],
     [ 2.,  0.,  0.],
     [ 0.,  1., -1.]]
scaler = MaxAbsScaler().fit(X) # scaler.max_abs_
X_transform = scaler.transform(X) 
scaler.inverse_transform(X_transform)
```
`[Preprocessing]: MinMaxScaler`
```python
from sklearn.preprocessing import MinMaxScaler

X = [[-1, 2],
     [-0.5, 6],
     [0, 10],
     [1, 18]]
scaler = MinMaxScaler()
scaler.fit(X)
X_transform = scaler.transform(X)
scaler.inverse_transform(X_transform)
```
`[Preprocessing]: StandardScaler`
```python
from sklearn.preprocessing import StandardScaler

X = [[0, 0],
     [0, 0],
     [1, 1],
     [1, 1]]
scaler = StandardScaler()
scaler.fit(X) # scaler.mean_, scaler.var_
X_transform = scaler.transform(X)
scaler.inverse_transform(X_transform)
```
`[Preprocessing]: RobustScaler`
```python
from sklearn.preprocessing import RobustScaler

X = [[ 1., -2.,  2.],
     [ -2.,  1.,  3.],
     [ 4.,  1., -2.]]
scaler = RobustScaler()
scaler.fit(X) # scaler.center_
X_transform = scaler.transform(X)
scaler.inverse_transform(X_transform)
```


### Datasets
#### Datasets: Real-World
`[Datasets]: load_breast_cancer`
```python
import pandas as pd
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()
X_data = dataset['data']
X_columns = dataset['feature_names']
y_data = dataset['target']
y_instances = dataset['target_names']

X = pd.DataFrame(data=X_data, columns=X_columns)
y = pd.DataFrame(data=y_data, columns=['target']) # instances: y_instances
dataset = pd.concat([X, y], axis=1)
dataset
```
`[Datasets]: load_diabetes`
```python
import pandas as pd
from sklearn.datasets import load_diabetes

dataset = load_diabetes()
X_data = dataset['data']
X_columns = dataset['feature_names']
y_data = dataset['target']

X = pd.DataFrame(data=X_data, columns=X_columns)
y = pd.DataFrame(data=y_data, columns=['target'])
dataset = pd.concat([X, y], axis=1)
dataset
```

`[Datasets]: load_digits`
```python
import pandas as pd
from sklearn.datasets import load_digits

dataset = load_digits()
X_data = dataset['data']
X_columns = dataset['feature_names']
y_data = dataset['target']
y_instances = dataset['target_names']

X = pd.DataFrame(data=X_data, columns=X_columns)
y = pd.DataFrame(data=y_data, columns=['target']) # y_instances
dataset = pd.concat([X, y], axis=1)
dataset
```
`[Datasets]: load_iris`
```python
import pandas as pd
from sklearn.datasets import load_iris

dataset = load_iris()
X_data = dataset['data']
X_columns = dataset['feature_names']
y_data = dataset['target']
y_instances = dataset['target_names']

X = pd.DataFrame(data=X_data, columns=X_columns)
y = pd.DataFrame(data=y_data, columns=['target']) # y_instances
dataset = pd.concat([X, y], axis=1)
dataset
```
`[Datasets]: load_wine`
```python
import pandas as pd
from sklearn.datasets import load_wine

dataset = load_wine()
X_data = dataset['data']
X_columns = dataset['feature_names']
y_data = dataset['target']
y_instances = dataset['target_names']

X = pd.DataFrame(data=X_data, columns=X_columns)
y = pd.DataFrame(data=y_data, columns=['target']) # y_instances
dataset = pd.concat([X, y], axis=1)
dataset
```


#### Datasets: Samples Generator 
`[Datasets]: make_blobs`
```python
import pandas as pd
from sklearn.datasets import make_blobs

X, y = make_blobs(
    n_samples=100,
    n_features=3, 
    centers=10,                # related with target
    cluster_std=1.0,           # related with feature
    center_box=(- 10.0, 10.0), # related with feature
    shuffle=True,
    random_state=None,
    return_centers=False)

data = dict()
for i in range(X.shape[1]):
    feature_name = 'feature_'+str(i)
    data[feature_name] = X[:, i].tolist()
data['target'] = y.tolist()
dataset = pd.DataFrame(data=data)
dataset
```
`[Datasets]: make_classification`
```python
import pandas as pd
from sklearn.datasets import make_classification

# n_features >= n_informative-n_redundant-n_repeated
# n_classes*n_clusters_per_class =< 2**n_informative
X, y = make_classification(
    n_samples=100, 
    n_features=20, 
    n_informative=2, 
    n_redundant=2, 
    n_repeated=0, 
    n_classes=2, 
    n_clusters_per_class=2, 
    weights=None, 
    flip_y=0.01, 
    class_sep=1.0, 
    hypercube=True, 
    shift=0.0, 
    scale=1.0, 
    shuffle=True, 
    random_state=None)

data = dict()
for i in range(X.shape[1]):
    feature_name = 'feature_'+str(i)
    data[feature_name] = X[:, i].tolist()
data['target'] = y.tolist()
dataset = pd.DataFrame(data=data)
dataset
```
`[Datasets]: make_multilabel_classification`
```python
import pandas as pd
from sklearn.datasets import make_multilabel_classification

X, y = make_multilabel_classification(
    n_samples=100, 
    n_features=10, 
    n_classes=5, 
    n_labels=2, 
    length=50, 
    allow_unlabeled=True, 
    sparse=False, 
    return_indicator='dense', 
    return_distributions=False, 
    random_state=None)

data = dict()
for i in range(X.shape[1]):
    feature_name = 'feature_'+str(i)
    data[feature_name] = X[:, i].tolist()
    
if y.ndim == 1:
    data['target_0'] = y.tolist()
else:     
    for i in range(y.shape[1]):
        feature_name = 'target_'+str(i)
        data[feature_name] = y[:, i].tolist()

dataset = pd.DataFrame(data=data)
dataset
```
`[Datasets]: make_regression`
```python
import pandas as pd
from sklearn.datasets import make_regression

X, y = make_regression(
    n_samples=100, 
    n_features=10, 
    n_informative=5, 
    n_targets=3, 
    bias=0.0, 
    effective_rank=None, 
    tail_strength=0.5, 
    noise=0.0, 
    shuffle=True, 
    coef=False, 
    random_state=None)

data = dict()
for i in range(X.shape[1]):
    feature_name = 'feature_'+str(i)
    data[feature_name] = X[:, i].tolist()

if y.ndim == 1:
    data['target_0'] = y.tolist()
else:     
    for i in range(y.shape[1]):
        feature_name = 'target_'+str(i)
        data[feature_name] = y[:, i].tolist()
    
dataset = pd.DataFrame(data=data)
dataset
```
`[Datasets]: make_gaussian_quantiles`
```python
import pandas as pd
from sklearn.datasets import make_gaussian_quantiles

X, y = make_gaussian_quantiles(
    mean=None, 
    cov=1.0, 
    n_samples=100, 
    n_features=2, 
    n_classes=3, 
    shuffle=True, 
    random_state=None)

data = dict()
for i in range(X.shape[1]):
    feature_name = 'feature_'+str(i)
    data[feature_name] = X[:, i].tolist()
data['target'] = y.tolist()
dataset = pd.DataFrame(data=data)
dataset
```
`[Datasets]: make_moons`
```python
import pandas as pd
from sklearn.datasets import make_moons

X, y = make_moons(
    n_samples=100,
    shuffle=True, 
    noise=None, 
    random_state=None)

data = dict()
for i in range(X.shape[1]):
    feature_name = 'feature_'+str(i)
    data[feature_name] = X[:, i].tolist()
data['target'] = y.tolist()
dataset = pd.DataFrame(data=data)
dataset
```
`[Datasets]: make_circles`
```python
import pandas as pd
from sklearn.datasets import make_circles

X, y = make_circles(
    n_samples=100, 
    shuffle=True, 
    noise=None, 
    random_state=None, 
    factor=0.8)

data = dict()
for i in range(X.shape[1]):
    feature_name = 'feature_'+str(i)
    data[feature_name] = X[:, i].tolist()
data['target'] = y.tolist()
dataset = pd.DataFrame(data=data)
dataset
```
`[Datasets]: make_sparse_uncorrelated`
```python
import pandas as pd
from sklearn.datasets import make_sparse_uncorrelated

X, y = make_sparse_uncorrelated(
    n_samples=100, 
    n_features=10, 
    random_state=None)

data = dict()
for i in range(X.shape[1]):
    feature_name = 'feature_'+str(i)
    data[feature_name] = X[:, i].tolist()
data['target'] = y.tolist()
dataset = pd.DataFrame(data=data)
dataset
```

### Metrics
#### Metrics: Classification
`Metric Entities`
```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0)
classifier = LogisticRegression()
classifier.fit(X, y)

data = dict()
data['y_true'] = y 
proba = classifier.predict_proba(X)
data['N_prob'] = proba[:, 0]
data['P_prob'] = proba[:, 1]
data['y_conf'] = classifier.decision_function(X)
data['y_pred'] = classifier.predict(X)
dataset = pd.DataFrame(data)

dataset['TP'] = dataset.y_true.mask((dataset.y_true == 1)&(dataset.y_pred == 1), '_MARKER_')
dataset['TP'] = dataset.TP.where(dataset.TP == '_MARKER_', False).astype(bool)
dataset['TN'] = dataset.y_true.mask((dataset.y_true == 0)&(dataset.y_pred == 0), '_MARKER_')
dataset['TN'] = dataset.TN.where(dataset.TN == '_MARKER_', False).astype(bool)
dataset['FP'] = dataset.y_true.mask((dataset.y_true == 0)&(dataset.y_pred == 1), '_MARKER_')
dataset['FP'] = dataset.FP.where(dataset.FP == '_MARKER_', False).astype(bool)
dataset['FN'] = dataset.y_true.mask((dataset.y_true == 1)&(dataset.y_pred == 0), '_MARKER_')
dataset['FN'] = dataset.FN.where(dataset.FN == '_MARKER_', False).astype(bool)

dataset['prediction-diagnosis'] = np.nan
dataset['prediction-diagnosis'] = dataset['prediction-diagnosis'].mask((dataset.TP == True), 'TP')
dataset['prediction-diagnosis'] = dataset['prediction-diagnosis'].mask((dataset.TN == True), 'TN')
dataset['prediction-diagnosis'] = dataset['prediction-diagnosis'].mask((dataset.FP == True), 'FP')
dataset['prediction-diagnosis'] = dataset['prediction-diagnosis'].mask((dataset.FN == True), 'FN')
dataset
```

`Confusion Matrix` : https://en.wikipedia.org/wiki/Confusion_matrix
```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0)
classifier = LogisticRegression()
classifier.fit(X, y)

data = dict()
data['y_true'] = y 
data['y_pred'] = classifier.predict(X)
dataset = pd.DataFrame(data)

print(classification_report(dataset['y_true'], dataset['y_pred']))
confusion_matrix = confusion_matrix(dataset['y_true'], dataset['y_pred'])
#recall = confusion_matrix[1, 1]/(confusion_matrix[1, 0]+confusion_matrix[1, 1])
#fallout = confusion_matrix[0, 1]/(confusion_matrix[0, 0]+confusion_matrix[0, 1])
confusion_matrix
```

`Matrics base on confusion matrix`
```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, jaccard_score, accuracy_score, balanced_accuracy_score, recall_score, precision_score, matthews_corrcoef, f1_score, fbeta_score, log_loss, brier_score_loss
from sklearn.metrics import classification_report

X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0)
classifier = LogisticRegression()
classifier.fit(X, y)

data = dict()
data['y_true'] = y
proba = classifier.predict_proba(X)
data['N_prob'] = proba[:, 0]
data['P_prob'] = proba[:, 1]
data['y_pred'] = classifier.predict(X)
dataset = pd.DataFrame(data)

print(classification_report(dataset['y_true'], dataset['y_pred']))
metrics = dict()
metrics['cohen_kappa_score'] = cohen_kappa_score(dataset['y_true'], dataset['y_pred'])
metrics['jaccard_score'] = jaccard_score(dataset['y_true'], dataset['y_pred'], average='binary')
metrics['accuracy'] = accuracy_score(dataset['y_true'], dataset['y_pred'])
metrics['balanced_accuracy_score'] = balanced_accuracy_score(dataset['y_true'], dataset['y_pred'])
metrics['precision'] = precision_score(dataset['y_true'], dataset['y_pred'], average='binary')
metrics['recall'] = recall_score(dataset['y_true'], dataset['y_pred'], average='binary')
metrics['f1'] = f1_score(dataset['y_true'], dataset['y_pred'], average='binary')
metrics['fbeta_score'] = fbeta_score(dataset['y_true'], dataset['y_pred'], beta=1, average='binary')
metrics['matthews_corrcoef'] = matthews_corrcoef(dataset['y_true'], dataset['y_pred'])

metrics['log_loss'] = log_loss(dataset['y_true'], dataset[['N_prob', 'P_prob']])
metrics['brier_score_loss'] = brier_score_loss(dataset['y_true'], dataset['P_prob'])
metrics
```

`ROC & AUC`
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0)
classifier = LogisticRegression()
classifier.fit(X, y)

data = dict()
data['y_true'] = y 
proba = classifier.predict_proba(X)
data['N_prob'] = proba[:, 0]
data['P_prob'] = proba[:, 1]
data['y_conf'] = classifier.decision_function(X)
data['y_pred'] = classifier.predict(X)
dataset = pd.DataFrame(data)

print(classification_report(dataset['y_true'], dataset['y_pred']))
confusion_matrix = confusion_matrix(dataset['y_true'], dataset['y_pred'])
recall = confusion_matrix[1, 1]/(confusion_matrix[1, 0]+confusion_matrix[1, 1])
fallout = confusion_matrix[0, 1]/(confusion_matrix[0, 0]+confusion_matrix[0, 1])
fpr, tpr, thresholds = roc_curve(dataset['y_true'], dataset['y_conf']) # or roc_curve(dataset['y_true'], dataset['P_prob'])

# visualization
print('- AUC:', auc(fpr, tpr))
plt.plot(fpr, tpr, 'o-') # X-axis(fpr): fall-out / y-axis(tpr): recall
plt.plot([fallout], [recall], 'ro', ms=10)
plt.plot([0, 1], [0, 1], 'k--')
plt.show()
```

`Integrated Evaluation`
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, jaccard_score, accuracy_score, balanced_accuracy_score, recall_score, precision_score, matthews_corrcoef, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0)
classifier = LogisticRegression()
classifier.fit(X, y)

data = dict()
data['y_true'] = y 
proba = classifier.predict_proba(X)
data['N_prob'] = proba[:, 0]
data['P_prob'] = proba[:, 1]
data['y_conf'] = classifier.decision_function(X)
data['y_pred'] = classifier.predict(X)
dataset = pd.DataFrame(data)

print(classification_report(dataset['y_true'], dataset['y_pred']))
metrics = dict()
metrics['cohen_kappa_score'] = cohen_kappa_score(dataset['y_true'], dataset['y_pred'])
metrics['jaccard_score'] = jaccard_score(dataset['y_true'], dataset['y_pred'], average='binary')
metrics['accuracy'] = accuracy_score(dataset['y_true'], dataset['y_pred'])
metrics['balanced_accuracy_score'] = balanced_accuracy_score(dataset['y_true'], dataset['y_pred'])
metrics['precision'] = precision_score(dataset['y_true'], dataset['y_pred'], average='binary')
metrics['recall'] = recall_score(dataset['y_true'], dataset['y_pred'], average='binary')
metrics['f1'] = f1_score(dataset['y_true'], dataset['y_pred'], average='binary')
metrics['fbeta_score'] = fbeta_score(dataset['y_true'], dataset['y_pred'], beta=1, average='binary')
metrics['matthews_corrcoef'] = matthews_corrcoef(dataset['y_true'], dataset['y_pred'])

confusion_matrix = confusion_matrix(dataset['y_true'], dataset['y_pred'])
recall = confusion_matrix[1, 1]/(confusion_matrix[1, 0]+confusion_matrix[1, 1])
fallout = confusion_matrix[0, 1]/(confusion_matrix[0, 0]+confusion_matrix[0, 1])
fpr, tpr, thresholds = roc_curve(dataset['y_true'], dataset['y_conf']) # or roc_curve(dataset['y_true'], dataset['P_prob'])

# visualization
print('- AUC:', auc(fpr, tpr))
plt.plot(fpr, tpr, 'o-') # X-axis(fpr): fall-out / y-axis(tpr): recall
plt.plot([fallout], [recall], 'ro', ms=10)
plt.plot([0, 1], [0, 1], 'k--')
plt.show()
```

#### Metrics: Regression


#### Metrics: Clustering



### Pipeline
- https://scikit-learn.org/stable/modules/compose.html
- https://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline


