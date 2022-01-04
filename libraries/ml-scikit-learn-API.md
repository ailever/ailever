https://scikit-learn.org/stable/modules/classes.html

## Sklearn BaseClass
### Mixin
```
- BaseEstimator
    - get_params
    - set_params
- ClassifierMixin
    - score
    - fit_predict
- RegressorMixin
    - score
- ClusterMixin
- BiclusterMixin
    - biclusters_
    - get_indices_
    - get_shape
    - get_submatrix
- TransformerMixin
    - fit_transform
- DensityMixin
    - score
- OutlierMixin
    - fit_predict
- MetaEstimatorMixin
- MultiOutputMixin
```

### APIs of scikit-learn objects
```
- ColumnTransformer
    - fit
    - fit_transform
    - get_feature_names
    - get_params
    - named_transformers_
    - set_params
    - transform
- FeatureUnion
    - fit
    - fit_transform
    - get_feature_names
    - get_params
    - n_features_in_
    - set_params
    - transform
- Pipeline
    - classes_
    - decision_function
    - fit
    - fit_predict
    - fit_transform
    - get_params
    - inverse_transform
    - n_features_in_
    - named_steps
    - predict
    - predict_log_proba
    - predict_proba
    - score
    - score_samples
    - set_params
    - transform
```

#### Classifier
```python
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances

class TemplateClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)
        # Store the classes seen during fit
        self.classes_ = unique_labels(y)

        self.X_ = X
        self.y_ = y
        # Return the classifier
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]
```

#### Regressor
```python
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class TemplateRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param

    def fit(self, X, y):
        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        self.X_ = X
        self.y_ = y
        # Return the regressor
        return self

    def predict(self, X):
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)
        return self.y_[slice(0, X.shape[0])]
```

#### Transformer
```python
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class TemplateTransformer(BaseEstimator, TransformerMixin):
    """This estimator ignores its input and returns random Gaussian noise.

    It also does not adhere to all scikit-learn conventions,
    but showcases how to handle randomness.
    """

    def __init__(self, n_components=100, random_state=None):
        self.random_state = random_state
        self.n_components = n_components

    # the arguments are ignored anyway, so we make them optional
    def fit(self, X=None, y=None):
        self.random_state_ = check_random_state(self.random_state)

    def transform(self, X):
        n_samples = X.shape[0]
        return self.random_state_.randn(n_samples, self.n_components)
```




---

## Preprocessing
```python
from sklearn import preprocessing
#preprocessing.add_dummy_feature(X)
#preprocessing.binarize(X)
#preprocessing.label_binarize(y)
#preprocessing.maxabs_scale(X)
#preprocessing.minmax_scale(X)
#preprocessing.normalize(X)
#preprocessing.quantile_transform(X)
#preprocessing.robust_scale(X)
#preprocessing.scale(X)
#preprocessing.power_transform(X)
preprocessing.Binarizer()
preprocessing.FunctionTransformer()
preprocessing.KBinsDiscretizer()
preprocessing.KernelCenterer()
preprocessing.LabelBinarizer()
preprocessing.LabelEncoder()
preprocessing.MultiLabelBinarizer()
preprocessing.MaxAbsScaler()
preprocessing.MinMaxScaler()
preprocessing.Normalizer()
preprocessing.OneHotEncoder()
preprocessing.OrdinalEncoder()
preprocessing.PolynomialFeatures()
preprocessing.PowerTransformer()
preprocessing.QuantileTransformer()
preprocessing.RobustScaler()
preprocessing.StandardScaler()

from sklearn import model_selection
#model_selection.train_test_split(X, y)
#model_selection.LeavePGroupsOut(n_groups)
#model_selection.LeavePOut(p)
#model_selection.PredefinedSplit(test_fold)
model_selection.check_cv()
model_selection.GroupKFold()
model_selection.GroupShuffleSplit()
model_selection.KFold()
model_selection.LeaveOneGroupOut()
model_selection.LeaveOneOut()
model_selection.RepeatedKFold()
model_selection.RepeatedStratifiedKFold()
model_selection.ShuffleSplit()
model_selection.StratifiedKFold()
model_selection.StratifiedShuffleSplit()
model_selection.TimeSeriesSplit()

from sklearn import manifold
#manifold.locally_linear_embedding(X)
#manifold.smacof(dissimilarities)
#manifold.spectral_embedding(adjacency)
#manifold.trustworthiness(X, X_embedded)
manifold.Isomap()
manifold.LocallyLinearEmbedding()
manifold.MDS()
manifold.SpectralEmbedding()
manifold.TSNE()

from sklearn import mixture
mixture.BayesianGaussianMixture()
mixture.GaussianMixture()

from sklearn import gaussian_process
#gaussian_process.kernels.CompoundKernel(kernels)
#gaussian_process.kernels.Exponentiation(kernel, exponent)
#gaussian_process.kernels.Hyperparameter(name, value_type, bounds)
#gaussian_process.kernels.Kernel()
#gaussian_process.kernels.Product(k1, k2)
#gaussian_process.kernels.Sum(k1, k2)
gaussian_process.kernels.ConstantKernel()
gaussian_process.kernels.DotProduct()
gaussian_process.kernels.ExpSineSquared()
gaussian_process.kernels.Matern()
gaussian_process.kernels.PairwiseKernel()
gaussian_process.kernels.RBF()
gaussian_process.kernels.RationalQuadratic()
gaussian_process.kernels.WhiteKernel()

from sklearn import feature_selection
#feature_selection.SelectFromModel(estimator)
#feature_selection.SequentialFeatureSelector(estimator)
#feature_selection.RFE(estimator)
#feature_selection.RFECV(estimator)
#feature_selection.chi2(X, y)
#feature_selection.f_classif(X, y)
#feature_selection.f_regression(X, y)
#feature_selection.r_regression(X, y)
#feature_selection.mutual_info_classif(X, y, *)
#feature_selection.mutual_info_regression(X, y, *)
feature_selection.GenericUnivariateSelect()
feature_selection.SelectPercentile()
feature_selection.SelectKBest()
feature_selection.SelectFpr()
feature_selection.SelectFdr()
feature_selection.SelectFwe()
feature_selection.VarianceThreshold()

from sklearn import feature_extraction
#feature_extraction.image.extract_patches_2d(image, patch_size)
#feature_extraction.image.grid_to_graph(n_x, n_y)
#feature_extraction.image.img_to_graph(img)
#feature_extraction.image.reconstruct_from_patches_2d(patches, image_size)
feature_extraction.DictVectorizer()
feature_extraction.FeatureHasher()
feature_extraction.image.PatchExtractor()
feature_extraction.text.CountVectorizer()
feature_extraction.text.HashingVectorizer()
feature_extraction.text.TfidfTransformer()
feature_extraction.text.TfidfVectorizer()

from sklearn import decomposition
#decomposition.SparseCoder(dictionary)
#decomposition.dict_learning(X, n_components)
#decomposition.dict_learning_online(X)
#decomposition.fastica(X)
#decomposition.non_negative_factorization(X)
#decomposition.sparse_encode(X, dictionary)
decomposition.DictionaryLearning()
decomposition.FactorAnalysis()
decomposition.FastICA()
decomposition.IncrementalPCA()
decomposition.KernelPCA()
decomposition.LatentDirichletAllocation()
decomposition.MiniBatchDictionaryLearning()
decomposition.MiniBatchSparsePCA()
decomposition.NMF()
decomposition.PCA()
decomposition.SparsePCA()
decomposition.TruncatedSVD()

from sklearn import cluster
#cluster.affinity_propagation(S)
#cluster.cluster_optics_dbscan(reachability, core_distances, ordering, eps)
#cluster.cluster_optics_xi(reachability, predecessor, ordering, min_samples)
#cluster.compute_optics_graph(X)
#cluster.dbscan(X)
#cluster.estimate_bandwidth(X)
#cluster.k_means(X, n_clusters)
#cluster.kmeans_plusplus(X, n_clusters)
#cluster.mean_shift(X)
#cluster.spectral_clustering(affinity)
#cluster.ward_tree(X)
cluster.AffinityPropagation()
cluster.AgglomerativeClustering()
cluster.Birch()
cluster.DBSCAN()
cluster.FeatureAgglomeration()
cluster.KMeans()
cluster.MiniBatchKMeans()
cluster.MeanShift()
cluster.OPTICS()
cluster.SpectralClustering()
cluster.SpectralBiclustering()
cluster.SpectralCoclustering()

from sklearn import covariance
#covariance.empirical_covariance(X)
#covariance.graphical_lasso(emp_cov, alpha)
#covariance.ledoit_wolf(X)
#covariance.oas(X)
#covariance.shrunk_covariance(emp_cov)
covariance.EmpiricalCovariance()
covariance.EllipticEnvelope()
covariance.GraphicalLasso()
covariance.GraphicalLassoCV()
covariance.LedoitWolf()
covariance.MinCovDet()
covariance.OAS()
covariance.ShrunkCovariance()

from sklearn import compose
#compose.ColumnTransformer(transformers)
#compose.make_column_transformer(*transformers)
compose.TransformedTargetRegressor()
compose.make_column_selector()
```

---

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

from sklearn import discriminant_analysis
discriminant_analysis.LinearDiscriminantAnalysis()
discriminant_analysis.QuadraticDiscriminantAnalysis()

from sklearn import gaussian_process
gaussian_process.GaussianProcessClassifier()
gaussian_process.GaussianProcessRegressor()

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


---


## Evaluation
```python
from sklearn import metrics

## CLASSIFICATION
metrics.accuracy_score(y_true, y_pred)
metrics.auc(x, y)
metrics.average_precision_score(y_true, y_pred)
metrics.balanced_accuracy_score(y_true, y_pred)
metrics.brier_score_loss(y_true, y_prob)
metrics.classification_report(y_true, y_pred)
metrics.cohen_kappa_score(y1, y2)
metrics.confusion_matrix(y_true, y_pred)
metrics.dcg_score(y_true, y_score)
metrics.det_curve(y_true, y_score)
metrics.f1_score(y_true, y_pred)
metrics.fbeta_score(y_true, y_pred, beta)
metrics.hamming_loss(y_true, y_pred)
metrics.hinge_loss(y_true, pred_decision)
metrics.jaccard_score(y_true, y_pred)
metrics.log_loss(y_true, y_pred)
metrics.matthews_corrcoef(y_true, y_pred)
metrics.multilabel_confusion_matrix(y_true, y_pred)
metrics.ndcg_score(y_true, y_score)
metrics.precision_recall_curve(y_true, y_pred)
metrics.precision_recall_fscore_support()
metrics.precision_score(y_true, y_pred)
metrics.recall_score(y_true, y_pred)
metrics.roc_auc_score(y_true, y_score)
metrics.roc_curve(y_true, y_score)
metrics.top_k_accuracy_score(y_true, y_score)
metrics.zero_one_loss(y_true, y_pred)

# REGRESSION
metrics.explained_variance_score(y_true, y_pred)
metrics.max_error(y_true, y_pred)
metrics.mean_absolute_error(y_true, y_pred)
metrics.mean_squared_error(y_true, y_pred)
metrics.mean_squared_log_error(y_true, y_pred)
metrics.median_absolute_error(y_true, y_pred)
metrics.mean_absolute_percentage_error()
metrics.r2_score(y_true, y_pred)
metrics.mean_poisson_deviance(y_true, y_pred)
metrics.mean_gamma_deviance(y_true, y_pred)
metrics.mean_tweedie_deviance(y_true, y_pred)
metrics.d2_tweedie_score(y_true, y_pred)
metrics.mean_pinball_loss(y_true, y_pred)
```

### Classification: Confusion Matrix
```python
from ailever.analysis import Evaluation
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

X, y = make_classification(n_samples=300, n_features=8, n_informative=5, n_redundant=1, n_repeated=1, n_classes=5, n_clusters_per_class=1, weights=[1/10, 3/10, 2/10, 1/10, 3/10])
classifier = LogisticRegression()
classifier.fit(X, y)

y_true = y 
y_pred = classifier.predict(X)
Evaluation.target_class_evaluation(y_true, y_pred)
```

### Classification: Binary-class ROC & AUC
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc

X, y = make_classification(n_samples=100, n_features=5, n_informative=2, n_redundant=1, n_repeated=1, n_classes=2, n_clusters_per_class=1)
classifier = LogisticRegression()
classifier.fit(X, y)

y_true = y 
y_prob = classifier.predict_proba(X)
y_pred = classifier.predict(X)

confusion_matrix = confusion_matrix(y_true, y_pred)
recall = confusion_matrix[1, 1]/(confusion_matrix[1, 0]+confusion_matrix[1, 1])
fallout = confusion_matrix[0, 1]/(confusion_matrix[0, 0]+confusion_matrix[0, 1])
fpr, tpr, thresholds = roc_curve(y_true, y_prob[:,1])

# visualization
print('- AUC:', auc(fpr, tpr))
plt.plot(fpr, tpr, 'o-') # X-axis(fpr): fall-out / y-axis(tpr): recall
plt.plot([fallout], [recall], 'bo', ms=10)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Fall-Out')
plt.ylabel('Recall')
plt.show()
```

### Classification: Multi-class ROC & AUC
```python
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

def multiclass_roc_curve(y_true, y_prob):
    import numpy as np
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    y_enco = label_binarize(y_true, np.sort(np.unique(y_true)).tolist())

    fpr = dict()
    tpr = dict()
    thr = dict()
    auc_ = dict()
    for class_idx in range(np.unique(y_true).shape[0]):
        fpr[class_idx], tpr[class_idx], thr[class_idx] = roc_curve(y_enco[:, class_idx], y_prob[:, class_idx])
        auc_[class_idx] = auc(fpr[class_idx], tpr[class_idx])
    return fpr, tpr, thr, auc_

X, y = make_classification(n_samples=300, n_features=8, n_informative=5, n_redundant=1, n_repeated=1, n_classes=5, n_clusters_per_class=1, weights=[1/10, 3/10, 2/10, 1/10, 3/10])
classifier = LogisticRegression()
classifier.fit(X, y)

y_true = y 
y_prob = classifier.predict_proba(X)
fpr, tpr, thr, auc = multiclass_roc_curve(y_true, y_prob)

for class_idx in range(np.unique(y_true).shape[0]):
    plt.plot(fpr[class_idx], tpr[class_idx], 'o-', ms=5, label=str(class_idx) + f' | {round(auc[class_idx], 2)}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Fall-Out')
plt.ylabel('Recall')
plt.legend()
```

### Classification: Binary-class PR & AUC
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_curve, auc

X, y = make_classification(n_samples=100, n_features=5, n_informative=2, n_redundant=1, n_repeated=1, n_classes=2, n_clusters_per_class=1)
classifier = LogisticRegression()
classifier.fit(X, y)

y_true = y 
y_prob = classifier.predict_proba(X)
y_pred = classifier.predict(X)

confusion_matrix = confusion_matrix(y_true, y_pred)
recall = confusion_matrix[1, 1]/(confusion_matrix[1, 0]+confusion_matrix[1, 1])
precision = confusion_matrix[0, 0]/(confusion_matrix[0, 0]+confusion_matrix[1, 0])
ppv, tpr, thresholds = precision_recall_curve(y_true, y_prob[:,1])

# visualization
print('- AUC:', auc(tpr, ppv))
plt.plot(tpr, ppv, 'o-') # X-axis(tpr): recall / y-axis(ppv): precision
plt.plot([recall], [precision], 'bo', ms=10)
plt.plot([0, 1], [1, 0], 'k--')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
```

### Classification: Multi-class PR & AUC
```python
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

def multiclass_pr_curve(y_true, y_prob):
    import numpy as np
    from sklearn.metrics import precision_recall_curve, auc
    from sklearn.preprocessing import label_binarize
    y_enco = label_binarize(y_true, np.sort(np.unique(y_true)).tolist())

    ppv = dict()
    tpr = dict()
    thr = dict()
    auc_ = dict()
    for class_idx in range(np.unique(y_true).shape[0]):
        ppv[class_idx], tpr[class_idx], thr[class_idx] = precision_recall_curve(y_enco[:, class_idx], y_prob[:, class_idx])
        auc_[class_idx] = auc(tpr[class_idx], ppv[class_idx])
    return ppv, tpr, thr, auc_


X, y = make_classification(n_samples=300, n_features=8, n_informative=5, n_redundant=1, n_repeated=1, n_classes=5, n_clusters_per_class=1, weights=[1/10, 3/10, 2/10, 1/10, 3/10])
classifier = LogisticRegression()
classifier.fit(X, y)

y_true = y 
y_prob = classifier.predict_proba(X)
ppv, tpr, thr, auc = multiclass_pr_curve(y_true, y_prob)

# visualization
for class_idx in range(np.unique(y_true).shape[0]):
    plt.plot(tpr[class_idx], ppv[class_idx], 'o-', ms=5, label=str(class_idx) + f' | {round(auc[class_idx], 2)}')
plt.plot([0, 1], [1, 0], 'k--')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
```

