- https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
- https://github.com/ailever/ailever/blob/master/libraries/ml-scikit-learn-API.md

## Classification
```python
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
```

---

## Regression
```python
```

---

## Linearity


