- https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
- https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn-API.md

## Classification
```python
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=300, n_features=8, n_informative=5, n_redundant=1, n_repeated=1, n_classes=5, n_clusters_per_class=1, weights=[1/10, 3/10, 2/10, 1/10, 3/10])
classifier = LogisticRegression()
classifier.fit(X, y)

y_true = y 
y_pred = classifier.predict(X)

## CLASSIFICATION
metrics.accuracy_score(y_true, y_pred)
metrics.balanced_accuracy_score(y_true, y_pred)
metrics.confusion_matrix(y_true, y_pred)
metrics.f1_score(y_true, y_pred, average='micro')
metrics.fbeta_score(y_true, y_pred, beta=2, average='micro')
metrics.hamming_loss(y_true, y_pred)
metrics.jaccard_score(y_true, y_pred, average='micro')
metrics.matthews_corrcoef(y_true, y_pred)
metrics.multilabel_confusion_matrix(y_true, y_pred)
metrics.precision_score(y_true, y_pred, average='micro')
metrics.recall_score(y_true, y_pred, average='micro')
metrics.zero_one_loss(y_true, y_pred)
```

### Binary-Class AUC
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, precision_recall_curve, auc

X, y = make_classification(n_samples=100, n_features=5, n_informative=2, n_redundant=1, n_repeated=1, n_classes=2, n_clusters_per_class=1)
classifier = LogisticRegression()
classifier.fit(X, y)

y_true = y 
y_prob = classifier.predict_proba(X)
y_pred = classifier.predict(X)

confusion_matrix = confusion_matrix(y_true, y_pred)
recall = confusion_matrix[1, 1]/(confusion_matrix[1, 0]+confusion_matrix[1, 1])
fallout = confusion_matrix[0, 1]/(confusion_matrix[0, 0]+confusion_matrix[0, 1])
precision = confusion_matrix[0, 0]/(confusion_matrix[0, 0]+confusion_matrix[1, 0])
fpr, tpr1, thresholds1 = roc_curve(y_true, y_prob[:,1])
ppv, tpr2, thresholds2 = precision_recall_curve(y_true, y_prob[:,1])

# visualization
print('- ROC AUC:', auc(fpr, tpr1))
print('- PR AUC:', auc(tpr2, ppv))
print(classification_report(y_true, y_pred, target_names=['down', 'up']))
plt.figure(figsize=(25, 7))
ax0 = plt.subplot2grid((1,2), (0,0))
ax1 = plt.subplot2grid((1,2), (0,1))

ax0.plot(fpr, tpr1, 'o-') # X-axis(fpr): fall-out / y-axis(tpr): recall
ax0.plot([fallout], [recall], 'bo', ms=10)
ax0.plot([0, 1], [0, 1], 'k--')
ax0.set_xlabel('Fall-Out')
ax0.set_ylabel('Recall')

ax1.plot(tpr2, ppv, 'o-') # X-axis(tpr): recall / y-axis(ppv): precision
ax1.plot([recall], [precision], 'bo', ms=10)
ax1.plot([0, 1], [1, 0], 'k--')
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
plt.show()
```

### Multi-Class AUC
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, precision_recall_curve, auc

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
y_pred = classifier.predict(X)
y_prob = classifier.predict_proba(X)
fpr, tpr1, thr1, auc1 = multiclass_roc_curve(y_true, y_prob)
ppv, tpr2, thr2, auc2 = multiclass_pr_curve(y_true, y_prob)

# visualization
print(classification_report(y_true, y_pred, target_names=['A', 'B', 'C', 'D', 'E']))
plt.figure(figsize=(25, 7))
ax0 = plt.subplot2grid((1,2), (0,0))
ax1 = plt.subplot2grid((1,2), (0,1))
for class_idx in range(np.unique(y_true).shape[0]):
    ax0.plot(fpr[class_idx], tpr1[class_idx], 'o-', ms=5, label=str(class_idx) + f' | {round(auc1[class_idx], 2)}')    
    ax1.plot(tpr2[class_idx], ppv[class_idx], 'o-', ms=5, label=str(class_idx) + f' | {round(auc2[class_idx], 2)}')

ax0.plot([0, 1], [0, 1], 'k--')
ax0.set_xlabel('Fall-Out')
ax0.set_ylabel('Recall')
ax0.legend() 
ax1.plot([0, 1], [1, 0], 'k--')
ax1.set_xlabel('Recall')
ax1.set_ylabel('Precision')
ax1.legend()
plt.show()
```

---

## Regression
```python
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=3000, n_features=10, n_informative=5, n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, noise=0.0, shuffle=True, coef=False, random_state=None)
regressor = LinearRegression()
regressor.fit(X, y)

y_true = y 
y_pred = regressor.predict(X)

# REGRESSION
metrics.explained_variance_score(y_true, y_pred)
metrics.max_error(y_true, y_pred)
metrics.mean_absolute_error(y_true, y_pred)
metrics.mean_squared_error(y_true, y_pred)
metrics.median_absolute_error(y_true, y_pred)
metrics.r2_score(y_true, y_pred)
metrics.mean_tweedie_deviance(y_true, y_pred)
```

---

## Linearity


