## [Machine Learning] | [scikit-learn](https://scikit-learn.org/stable/modules/classes.html#) | [github](https://github.com/scikit-learn/scikit-learn)

- [Supervised Learning: Classification](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#classification)
- [Supervised Learning: Regression](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#regression)
- [Unsupervised Learning](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#unsupervised-learning)
- [Feature Selection](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#feature-selection)
- [Model Selection](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#model-selection)
- [Preprocessing](https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn.md#preprocessing)


---

### Classification
#### Classification: linear_model
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
`[Classification]: BernoulliNB`
```python
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
dataset = SKAPI.boston(download=False)
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
dataset = SKAPI.boston(download=False)
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
dataset = SKAPI.boston(download=False)
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
dataset = SKAPI.boston(download=False)
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
dataset = SKAPI.boston(download=False)
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
dataset = SKAPI.boston(download=False)
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
dataset = SKAPI.boston(download=False)
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
dataset = SKAPI.boston(download=False)
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
dataset = SKAPI.boston(download=False)
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
dataset = SKAPI.boston(download=False)
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
dataset = SKAPI.boston(download=False)
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
dataset = SKAPI.boston(download=False)
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
dataset = SKAPI.boston(download=False)
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
dataset = SKAPI.boston(download=False)
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
dataset = SKAPI.boston(download=False)
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
dataset = SKAPI.boston(download=False)
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
dataset = SKAPI.boston(download=False)
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
dataset = SKAPI.boston(download=False)
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
dataset = SKAPI.boston(download=False)
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
dataset = SKAPI.boston(download=False)
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
dataset = SKAPI.boston(download=False)
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
`[Decomposing signals in components]: `
```python
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
`[Feature Selection]: SequentialFeatureSelector`
```python
from ailever.dataset import SKAPI
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SequentialFeatureSelector

dataset = SKAPI.iris(download=False)
X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

classifier = LassoCV().fit(X, y)
selector1 = SequentialFeatureSelector(classifier, n_features_to_select=2, direction="forward").fit(X,y)
selector2 = SequentialFeatureSelector(classifier, n_features_to_select=2, direction="forward").fit(X,y)

X_new1 = X[:, selector1.get_support()]
X_new2 = X[:, selector2.get_support()]
print(X.shape, X_new1.shape, X_new2.shape)
```



### Model Selection
#### Model Selection: Dataset-Spliter
`[Model Selection]: LeaveOneOut`
```python
from sklearn.model_selection import LeaveOneOut

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
     print("%s %s" % (train_index, test_index))
```
`[Model Selection]: KFold`
```python
from sklearn.model_selection import KFold

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]
kf = KFold(n_splits=3, shuffle=True)
for train_index, test_index in kf.split(X):
     print("%s %s" % (train_index, test_index))
```
`[Model Selection]: StratifiedKFold`
```python
from sklearn.model_selection import StratifiedKFold

X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
skf = StratifiedKFold(n_splits=4, shuffle=True)
for train_index, test_index in skf.split(X, y):
    print("%s %s" % (train_index, test_index))
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
tss = TimeSeriesSplit()
for train_index, test_index in tss.split(X):
     print("%s %s" % (train_index, test_index))
```

### Preprocessing
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


