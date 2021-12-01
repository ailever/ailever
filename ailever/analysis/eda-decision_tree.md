
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
print(iris.feature_names)
graphviz.Source(dot_data)
```
