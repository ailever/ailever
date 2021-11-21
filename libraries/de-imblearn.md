## Evaluation
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
    weights=[0.99, 0.01], 
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

TOTAL = dataset.shape[0]
P = dataset["y_true"].sum()
N = TOTAL - dataset["y_true"].sum()
PP = dataset["y_pred"].sum()
NN = TOTAL - dataset["y_pred"].sum()
TP = dataset['TP'].sum() # TP: when y_pred is 1, y_true is 1
FP = dataset['FP'].sum() # FP: when y_pred is 1, y_true is 0 > type2-error
TN = dataset['TN'].sum() # TN: when y_pred is 0, y_true is 0
FN = dataset['FN'].sum() # FN: when y_pred is 0, y_true is 1 > type1-error
confusion_matrix = np.array([[TP,FN],
                             [FP,TN]])
"""
DO NOT CONFUSE the denotation T, F of the arguments of P(~) below!
- Confusion matrix

                    y_pred(1):PP                              y_pred(0):NN                   summation
                                                                                             FP+FN    : P(PP|F) / P(NN|F)
y_true(1):P              TP                                        FN                        TP+FN    : recall=P(PP|P) / miss rate=P(NN|P)      
y_true(0):N              FP                                        TN                        FP+TN    : fall-out=P(PP|N) / selectivity=P(NN|N)
                                                                                             TP+TN    : P(PP|T) / P(NN|T)
summation              TP+FP                                     FN+TN            
                         ..                                        ..
              precision=P(T|PP)=P(P|PP)                false omission rate=P(F|NN)=P(P|NN)
         /false discovery rate=P(F|PP)=P(N|PP)     /negative predictive value=P(T|NN)=P(N|NN)
                   
                   

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
metrics['precision'] = (TP)/(TP+FP)           # precision_score(dataset['y_true'], dataset['y_pred'], average='binary')
metrics['false-discovery-rate'] = (FP)/(TP+FP)
metrics['false-omission-rate'] = (FN)/(FN+TN)
metrics['negative-predictive-value'] = (TN)/(FN+TN)
metrics['recall'] = (TP)/(TP+FN)              # recall_score(dataset['y_true'], dataset['y_pred'], average='binary')
metrics['miss-rate'] = (FN)/(TP+FN)
metrics['fall-out'] = (FP)/(FP+TN)
metrics['selectivity'] = (TN)/(FP+TN)

metrics['accuracy'] = (TP+TN)/(TP+TN+FP+FN)   # accuracy_score(dataset['y_true'], dataset['y_pred'])
metrics['prevalence'] = (TP+FN)/(TP+TN+FP+FN)
metrics['balanced-accuracy'] = (metrics['recall'] + metrics['selectivity'])/ 2

metrics['positive-likelihood-ratio'] = metrics['recall'] / metrics['fall-out']
metrics['negative-likelihood-ratio'] = metrics['miss-rate'] / metrics['selectivity']
metrics['f1'] =  (2*metrics['recall']*metrics['precision'])/(metrics['recall'] + metrics['precision'])  # f1_score(dataset['y_true'], dataset['y_pred'], average='binary')
metrics['Fowlkes–Mallows index'] = np.sqrt(metrics['recall']*metrics['precision'])
metrics['Matthews-correlation-coefficient'] = np.sqrt(metrics['recall']*metrics['selectivity']*metrics['precision']*metrics['negative-predictive-value']) - np.sqrt(metrics['miss-rate']*metrics['fall-out']*metrics['false-omission-rate']*metrics['false-discovery-rate'])

# denotation(1) based on conditional probability,
metrics['P(PP|P)'] = (TP)/(TP+FN) # sensitivity, recall, hit rate, or true positive rate (TPR)
metrics['P(NN|P)'] = (FN)/(TP+FN) # miss rate or false negative rate (FNR)
metrics['P(PP|N)'] = (FP)/(FP+TN) # fall-out or false positive rate (FPR)
metrics['P(NN|N)'] = (TN)/(FP+TN) # specificity, selectivity or true negative rate (TNR)
metrics['P(P|PP)'] = (TP)/(TP+FP) # precision or positive predictive value (PPV)
metrics['P(P|NN)'] = (FN)/(FN+TN) # false omission rate (FOR)
metrics['P(N|PP)'] = (FP)/(TP+FP) # false discovery rate (FDR)
metrics['P(N|NN)'] = (TN)/(FN+TN) # negative predictive value (NPV)

# denotation(2) in the event with "performance of classifier(T/F)" based on conditional probability,
metrics['P(PP|T)'] = TP/(TP+TN)  
metrics['P(PP|F)'] = FP/(FP+FN) # type2-error  
metrics['P(NN|T)'] = TN/(TP+TN) 
metrics['P(NN|F)'] = FN/(FP+FN) # type1-error
metrics['P(T|PP)'] = TP/(TP+FP) # precision or positive predictive value (PPV) = 1 - FDR
metrics['P(F|PP)'] = FP/(TP+FP) # false discovery rate (FDR) = 1 - PPV
metrics['P(T|NN)'] = TN/(FN+TN) # negative predictive value (NPV) = 1 - FOR
metrics['P(F|NN)'] = FN/(FN+TN) # false omission rate (FOR) = 1 - NPV

classifier_performance = dict()
classifier_performance['prediction_performance'] = dict()
classifier_performance['decision_performance'] = dict() 
classifier_performance['domain_adaptation_performance'] = dict() 

classifier_performance['prediction_performance']['accuracy'] = metrics['accuracy']
classifier_performance['prediction_performance']['balanced-accuracy'] = metrics['balanced-accuracy']
classifier_performance['prediction_performance']['recall'] = metrics['recall']
classifier_performance['prediction_performance']['selectivity'] = metrics['selectivity']

classifier_performance['decision_performance']['precision'] = metrics['precision']
classifier_performance['decision_performance']['negative-predictive-value'] = metrics['negative-predictive-value']
classifier_performance['decision_performance']['ture-positive-decision-ratio'] = metrics['P(PP|T)'] # cross-term : ture-positive-decision-ratio
classifier_performance['decision_performance']['ture-negative-decision-ratio'] = metrics['P(NN|T)'] # cross-term : ture-negative-decision-ratio
classifier_performance['decision_performance']['type1-error-ratio'] = metrics['P(NN|F)'] # cross-term : type1-error-ratio
classifier_performance['decision_performance']['type2-error-ratio'] = metrics['P(PP|F)'] # cross-term : type2-error-ratio

classifier_performance['domain_adaptation_performance']['Matthews-correlation-coefficient'] = metrics['Matthews-correlation-coefficient']
classifier_performance['domain_adaptation_performance']['f1'] = metrics['f1']
classifier_performance['domain_adaptation_performance']['Fowlkes–Mallows index'] = metrics['Fowlkes–Mallows index']

curiousity = dict()
curiousity['P(T|PP)'] = metrics['P(T|PP)'] # precision
curiousity['P(T|NN)'] = metrics['P(T|NN)'] # negative predictive value
curiousity['P(F|PP)'] = metrics['P(F|PP)'] # false discovery rate
curiousity['P(F|NN)'] = metrics['P(F|NN)'] # false omission rate

print(f'- y_true[1/0]: [{P}/{N}]', [P/TOTAL, N/TOTAL])
print(f'- y_pred[1/0]: [{PP}/{NN}]', [PP/TOTAL, NN/TOTAL])

print('\n* Classifier Performance')
for performance in classifier_performance.items():
    print(performance[0].upper())
    for item in performance[1].items():
        print(f'- {item[0]}:', item[1])    

print('\n* Curiousity')
for item in curiousity.items():
    print(f'- {item[0]}:', item[1])
dataset
```


## Over-Sampling

## Under-Sampling

## Combining Over-and Under-Sampling
