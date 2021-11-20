
```python
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn import linear_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# n_features >= n_informative-n_redundant-n_repeated
# n_classes*n_clusters_per_class =< 2**n_informative
X, y = make_classification(
    n_samples=10000, 
    n_features=5, 
    n_informative=2, 
    n_redundant=2, 
    n_repeated=0, 
    n_classes=2, 
    n_clusters_per_class=1, 
    weights=[0.9, 0.1], 
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


X = dataset.loc[:, dataset.columns != 'target'].values
y = dataset.loc[:, dataset.columns == 'target'].values.ravel()

# [STEP2]: model
classifier = linear_model.LogisticRegression(penalty='l2', max_iter=500)
classifier.fit(X, y)

# [STEP3]: save & load
joblib.dump(classifier, 'classifier.joblib')
classifier = joblib.load('classifier.joblib')

# [STEP4]: prediction
data = dict()
data['y_true'] = y 
data['y_pred'] = classifier.predict(X)

proba = classifier.predict_proba(X)
data['N_prob'] = proba[:, 0]
data['P_prob'] = proba[:, 1]

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

TOTAL = dataset.shape[0]
Actual_True = dataset["y_true"].sum()
Actual_False = TOTAL - dataset["y_true"].sum()
P = dataset["y_pred"].sum()
N = TOTAL - dataset["y_pred"].sum()
TP = dataset['TP'].sum() # TP: when y_pred is 1, y_true is 1
FP = dataset['FP'].sum() # FP: when y_pred is 1, y_true is 0 > type1-error
TN = dataset['TN'].sum() # TN: when y_pred is 0, y_true is 0
FN = dataset['FN'].sum() # FN: when y_pred is 0, y_true is 1 > type2-error
confusion_matrix = np.array([[TP,FN],
                             [FP,TN]])
"""
DO NOT CONFUSE the denotation T, F of the arguments of P(~) below!
- Confusion matrix

                y_pred(1)                         y_pred(0)             summation
                                                                          FP+FN    : P(P|F) / P(N|F)
y_true(1)          TP                                FN                   TP+FN    : recall / miss rate      
y_true(0)          FP                                TN                   FP+TN    : fall-out / selectivity
                                                                          TP+TN    : P(P|T) / P(N|T)
summation        TP+FP                             FN+TN            
                   ..                                ..
            precision=P(T|P)              false omission rate=P(F|N)
      /false discovery rate=P(F|P)     /negative predictive value=P(T|N)
                   
                   

* [classifier_performance]
- The P(P|T)=TP/(TP+TN) means probabilty that y_pred is 1 when the (y_true, y_pred) is in "(1,1), (0,0)" 
- The P(P|F)=FP/(FP+FN) means probabilty that y_pred is 1 when the (y_true, y_pred) is in "(1,0), (0,1)"
- The P(N|T)=TN/(TP+TN) means probabilty that y_pred is 0 when the (y_true, y_pred) is in "(1,1), (0,0)"
- The P(N|F)=FN/(FP+FN) means probabilty that y_pred is 0 when the (y_true, y_pred) is in "(1,0), (0,1)"

* [curiousity]
- The P(T|P)=TP/(TP+FP) means when the y_pred is 1, the y_true is 1
- The P(F|P)=FP/(TP+FP) means when the y_pred is 1, the y_true is 0
- The P(T|N)=TN/(FN+TN) means when the y_pred is 0, the y_true is 0
- The P(F|N)=FN/(FN+TN) means when the y_pred is 0, the y_true is 1
"""

metrics = dict()
metrics['accuracy'] = (TP+TN)/(TP+TN+FP+FN) # accuracy_score(dataset['y_true'], dataset['y_pred'])
metrics['precision'] = (TP)/(TP+FP)         # precision_score(dataset['y_true'], dataset['y_pred'], average='binary')
metrics['false-discovery-rate'] = (FP)/(TP+FP)
metrics['false-omission-rate'] = (FN)/(FN+TN)
metrics['negative-predictive-value'] = (TN)/(FN+TN)
metrics['recall'] = (TP)/(TP+FN)            # recall_score(dataset['y_true'], dataset['y_pred'], average='binary')
metrics['miss-rate'] = (FN)/(TP+FN)
metrics['fall-out'] = (FP)/(FP+TN)
metrics['selectivity'] = (TN)/(FP+TN)
metrics['f1'] = (2*TP)/(2*TP + FP + FN)     # f1_score(dataset['y_true'], dataset['y_pred'], average='binary')
metrics['P(P|T)'] = TP/(TP+TN)  
metrics['P(P|F)'] = FP/(FP+FN) 
metrics['P(N|T)'] = TN/(TP+TN) 
metrics['P(N|F)'] = FN/(FP+FN) 
metrics['P(T|P)'] = TP/(TP+FP) # precision or positive predictive value (PPV) = 1 - FDR
metrics['P(F|P)'] = FP/(TP+FP) # false discovery rate (FDR) = 1 - PPV
metrics['P(T|N)'] = TN/(FN+TN) # negative predictive value (NPV) = 1 - FOR
metrics['P(F|N)'] = FN/(FN+TN) # false omission rate (FOR) = 1 - NPV

classifier_performance = dict()
classifier_performance['accuracy'] = metrics['accuracy']
classifier_performance['precision'] = metrics['precision']
classifier_performance['recall'] = metrics['recall']
classifier_performance['selectivity'] = metrics['selectivity']
classifier_performance['f1'] = metrics['f1']

curiousity = dict()
curiousity['P(T|P)'] = metrics['P(T|P)']
curiousity['P(T|N)'] = metrics['P(T|N)'] 
curiousity['P(F|P)'] = metrics['P(F|P)']
curiousity['P(F|N)'] = metrics['P(F|N)']

print(f'- y_true[1/0]: [{Actual_True}/{Actual_False}]', [Actual_True/TOTAL, Actual_False/TOTAL])
print(f'- y_pred[1/0]: [{P}/{N}]', [P/TOTAL, N/TOTAL])

print('\n* Classifier Performance')
for item in classifier_performance.items():
    print(f'- {item[0]}:', item[1])    

print('\n* Curiousity')
for item in curiousity.items():
    print(f'- {item[0]}:', item[1])
dataset
```
