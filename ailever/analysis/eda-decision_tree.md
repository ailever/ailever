
```python
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

# define dataset
iris = load_iris()
X = iris.data
y = iris.target

criterions = ['gini', 'entropy']
splitters = ['best', 'random']
classifier = DecisionTreeClassifier(
    criterion=criterions[0],
    splitter=splitters[1],
    max_depth=None,
    min_samples_split=30,
    min_samples_leaf=30,
)
classifier.fit(X, y)

dot_data=export_graphviz(classifier,
                         out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True,
                         rounded=True,
                         special_characters=True)

print('- Features:', iris.feature_names)
print('- Target  :', iris.target_names)
graphviz.Source(dot_data)
```


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz

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

# define dataset
iris = load_iris()
X = iris.data
y = iris.target

criterions = ['gini', 'entropy']
splitters = ['best', 'random']
classifier = DecisionTreeClassifier(
    criterion=criterions[0],
    splitter=splitters[1],
    max_depth=None,
    min_samples_split=30,
    min_samples_leaf=30,
)
classifier.fit(X, y)

dot_data=export_graphviz(classifier,
                         out_file=None,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True,
                         rounded=True,
                         special_characters=True)

print('- Features:', iris.feature_names)
print('- Target  :', iris.target_names)
display(graphviz.Source(dot_data))


y_true = y 
y_pred = classifier.predict(X)
y_prob = classifier.predict_proba(X)
fpr, tpr1, thr1, auc1 = multiclass_roc_curve(y_true, y_prob)
ppv, tpr2, thr2, auc2 = multiclass_pr_curve(y_true, y_prob)

plt.figure(figsize=(25,7))
ax1 = plt.subplot2grid((1,2), (0,0))
ax2 = plt.subplot2grid((1,2), (0,1))

print(metrics.classification_report(y_true, y_pred))
for class_idx in range(np.unique(y_true).shape[0]):
    ax1.plot(fpr[class_idx], tpr1[class_idx], 'o-', ms=5, label=str(class_idx) + f' | {round(auc1[class_idx], 2)}')
    ax2.plot(tpr2[class_idx], ppv[class_idx], 'o-', ms=5, label=str(class_idx) + f' | {round(auc2[class_idx], 2)}')
ax1.plot([0, 1], [0, 1], 'k--')
ax2.plot([0, 1], [1, 0], 'k--')
ax1.set_xlabel('Fall-Out')
ax1.set_ylabel('Recall')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax1.legend()
ax2.legend()
plt.show()
```
![image](https://user-images.githubusercontent.com/56889151/149611386-a2ba852f-3b89-4c1a-abc2-f0c388010f65.png)
![image](https://user-images.githubusercontent.com/56889151/149611393-d7819576-4548-489a-a23b-f9b833b1380f.png)

