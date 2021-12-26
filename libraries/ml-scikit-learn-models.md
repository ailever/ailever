## Training
```python
from sklearn import linear_model
#linear_model.enet_path(X, y)
#linear_model.lars_path(X, y)
#linear_model.lars_path_gram(Xy, Gram)
#linear_model.lasso_path(X, y)
#linear_model.orthogonal_mp(X, y)
#linear_model.orthogonal_mp_gram(Gram, Xy)
#linear_model.ridge_regression(X, y, alpha)
linear_model.LogisticRegression()
linear_model.LogisticRegressionCV()
linear_model.PassiveAggressiveClassifier()
linear_model.Perceptron()
linear_model.RidgeClassifier()
linear_model.RidgeClassifierCV()
linear_model.SGDClassifier()
linear_model.ElasticNet()
linear_model.ElasticNetCV()
linear_model.Lars()
linear_model.LarsCV()
linear_model.Lasso()
linear_model.LassoCV()
linear_model.LassoLars()
linear_model.LassoLarsCV()
linear_model.LassoLarsIC()
linear_model.OrthogonalMatchingPursuit()
linear_model.OrthogonalMatchingPursuitCV()
linear_model.ARDRegression()
linear_model.BayesianRidge()
linear_model.MultiTaskElasticNet()
linear_model.MultiTaskElasticNetCV()
linear_model.MultiTaskLasso()
linear_model.MultiTaskLassoCV()
linear_model.HuberRegressor()
linear_model.RANSACRegressor()
linear_model.TheilSenRegressor()
linear_model.PoissonRegressor()
linear_model.TweedieRegressor()
linear_model.GammaRegressor()
linear_model.PassiveAggressiveRegressor()

from sklearn import ensemble
#ensemble.StackingClassifier(estimators)
#ensemble.StackingRegressor(estimators)
#ensemble.VotingClassifier(estimators)
#ensemble.VotingRegressor(estimators)
ensemble.AdaBoostClassifier()
ensemble.AdaBoostRegressor()
ensemble.BaggingClassifier()
ensemble.BaggingRegressor()
ensemble.ExtraTreesClassifier()
ensemble.ExtraTreesRegressor()
ensemble.GradientBoostingClassifier()
ensemble.GradientBoostingRegressor()
ensemble.IsolationForest()
ensemble.RandomForestClassifier()
ensemble.RandomForestRegressor()
ensemble.RandomTreesEmbedding()

from sklearn import naive_bayes
naive_bayes.BernoulliNB()
naive_bayes.CategoricalNB()
naive_bayes.ComplementNB()
naive_bayes.GaussianNB()
naive_bayes.MultinomialNB()

from sklearn import neighbors
#neighbors.BallTree(X[, leaf_size, metric])
#neighbors.KDTree(X[, leaf_size, metric])
#neighbors.kneighbors_graph(X, n_neighbors)
#neighbors.radius_neighbors_graph(X, radius)
neighbors.KernelDensity()
neighbors.KNeighborsClassifier()
neighbors.KNeighborsRegressor()
neighbors.KNeighborsTransformer()
neighbors.LocalOutlierFactor()
neighbors.RadiusNeighborsClassifier()
neighbors.RadiusNeighborsRegressor()
neighbors.RadiusNeighborsTransformer()
neighbors.NearestCentroid()
neighbors.NearestNeighbors()
neighbors.NeighborhoodComponentsAnalysis()

from sklearn import neural_network
neural_network.BernoulliRBM()
neural_network.MLPClassifier()
neural_network.MLPRegressor()

from sklearn import svm
svm.LinearSVC()
svm.LinearSVR()
svm.NuSVC()
svm.NuSVR()
svm.OneClassSVM()
svm.SVC()
svm.SVR()

from sklearn import tree
#tree.export_graphviz(decision_tree)
#tree.export_text(decision_tree)tree.DecisionTreeClassifier()
tree.DecisionTreeRegressor()
tree.ExtraTreeClassifier()
tree.ExtraTreeRegressor()

from sklearn import pipeline
#pipeline.FeatureUnion(transformer_list)
#pipeline.Pipeline(steps)
#pipeline.make_pipeline(*steps[, memory, verbose])
#pipeline.make_union(*transformers[, n_jobs, ...])

from sklearn import model_selection
#model_selection.GridSearchCV(estimator, param_grid)
#model_selection.ParameterGrid(param_grid)
#model_selection.ParameterSampler(param_distributions, n_iter)
#model_selection.RandomizedSearchCV(estimator, param_distributions)
```

## Preprocessing
```python
```
