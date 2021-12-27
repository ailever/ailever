import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

class Evaluation:
    def __init__(self):
        pass

    @staticmethod
    def classification(y_true, y_pred):
        pass

    @staticmethod
    def regression(y_true, y_pred):
        pass

    @staticmethod
    def clustering(y_true):
        pass
 
    @staticmethod
    def information_value(y_true, y_pred):
        pass
    
    @staticmethod
    def feature_importance(y_true, y_pred):
        pass

    @staticmethod
    def target_class_evaluation(y_true, y_pred):
        conf_matrix = confusion_matrix(y_true, y_pred)

        metrics = dict()
        for metric in ['P', 'N', 'PP', 'PN'] + ['TP', 'FP', 'TN', 'FN']+['PPV', 'FDR', 'FOR', 'NPV']+['TPR', 'FPR', 'FNR', 'TNR']+['LR+', 'LR-']+['MK', 'DOR']+['F1', 'FM']+['MCC', 'JI']+['ACC', 'Prevalence', 'BA', 'BM', 'PT']:
            metrics[metric] = dict()

        for true_idx in range(conf_matrix.shape[0]):
            metrics['P'][true_idx] = conf_matrix[true_idx, :].sum()
            metrics['N'][true_idx] = conf_matrix[:, :].sum() - conf_matrix[true_idx, :].sum()
            metrics['Prevalence'][true_idx] = metrics['P'][true_idx]/(metrics['P'][true_idx] + metrics['N'][true_idx])

        for pred_idx in range(conf_matrix.shape[1]):
            metrics['PP'][pred_idx] = conf_matrix[:, pred_idx].sum() # Positive
            metrics['PN'][pred_idx] = conf_matrix[:, :].sum() - conf_matrix[:, pred_idx].sum() # Negative
            metrics['TP'][pred_idx] = conf_matrix[pred_idx, pred_idx] # hit
            metrics['FP'][pred_idx] = conf_matrix[:, pred_idx].sum() - conf_matrix[pred_idx, pred_idx] # type I error, false alarm, overestimation
            metrics['TN'][pred_idx] = np.trace(conf_matrix[:, :]) - conf_matrix[pred_idx, pred_idx] # correct rejection
            metrics['FN'][pred_idx] = (conf_matrix[:, :].sum() - conf_matrix[:, pred_idx].sum()) - (np.trace(conf_matrix[:, :]) - conf_matrix[pred_idx, pred_idx]) # type II error, miss, underestimation

        for true_idx in range(conf_matrix.shape[0]):
            pred_idx = true_idx
            metrics['TPR'][true_idx] = metrics['TP'][pred_idx]/metrics['P'][true_idx] # True positive rate (TPR), recall, sensitivity (SEN), probability of detection, hit rate, power
            metrics['FPR'][true_idx] = metrics['FP'][pred_idx]/metrics['N'][true_idx] # False positive rate (FPR), probability of false alarm, fall-out
            metrics['FNR'][true_idx] = metrics['FN'][pred_idx]/metrics['P'][true_idx] # False negative rate (FNR), miss rate
            metrics['TNR'][true_idx] = metrics['TN'][pred_idx]/metrics['N'][true_idx] # True negative rate (TNR), specificity (SPC), selectivity
            metrics['ACC'][true_idx] = (metrics['TP'][pred_idx] + metrics['TN'][pred_idx])/(metrics['P'][true_idx] + metrics['N'][true_idx]) # Accuracy (ACC)
            metrics['BA'][true_idx] = metrics['TPR'][true_idx]/metrics['TNR'][true_idx] # Balanced accuracy (BA)
            metrics['BM'][true_idx] = metrics['TPR'][true_idx] + metrics['TNR'][true_idx] - 1 # Informedness, bookmaker informedness (BM)
            metrics['PT'][true_idx] = (np.sqrt(metrics['TPR'][true_idx]*metrics['FPR'][true_idx])-metrics['FPR'][true_idx])/(metrics['TPR'][true_idx] - metrics['FPR'][true_idx]) # Prevalence threshold (PT)

        for pred_idx in range(conf_matrix.shape[1]):
            true_idx = pred_idx
            metrics['PPV'][pred_idx] = metrics['TP'][pred_idx]/metrics['PP'][pred_idx] # Positive predictive value (PPV), precision
            metrics['FDR'][pred_idx] = metrics['FP'][pred_idx]/metrics['PP'][pred_idx] # False discovery rate (FDR)
            metrics['FOR'][pred_idx] = metrics['FN'][pred_idx]/metrics['PN'][pred_idx] # False omission rate (FOR)
            metrics['NPV'][pred_idx] = metrics['TN'][pred_idx]/metrics['PN'][pred_idx] # Negative predictive value (NPV)
            metrics['MK'][pred_idx] = metrics['PPV'][pred_idx] + metrics['NPV'][pred_idx] - 1 # Markedness (MK), deltaP (Δp)
            metrics['F1'][pred_idx] = (2*metrics['TP'][pred_idx])/(2*metrics['TP'][pred_idx] + metrics['FP'][pred_idx] + metrics['FN'][pred_idx]) # F1 score
            metrics['FM'][pred_idx] = np.sqrt(metrics['PPV'][pred_idx]*metrics['TPR'][true_idx]) # Fowlkes–Mallows index (FM)
            metrics['MCC'][pred_idx] = np.sqrt(metrics['TPR'][true_idx]*metrics['TNR'][true_idx]*metrics['PPV'][pred_idx]*metrics['NPV'][pred_idx]) - np.sqrt(metrics['FNR'][true_idx]*metrics['FPR'][true_idx]*metrics['FOR'][pred_idx]*metrics['FDR'][pred_idx]) # Matthews correlation coefficient (MCC)
            metrics['JI'][pred_idx] = metrics['TP'][pred_idx] / (metrics['TP'][pred_idx] + metrics['FN'][pred_idx] + metrics['FP'][pred_idx]) # Threat score (TS), critical success index (CSI), Jaccard index

        for true_idx in range(conf_matrix.shape[0]):
            metrics['LR+'][true_idx] = metrics['TPR'][true_idx]/metrics['FPR'][true_idx] # Positive likelihood ratio (LR+)
            metrics['LR-'][true_idx] = metrics['FNR'][true_idx]/metrics['TNR'][true_idx] # Negative likelihood ratio (LR−)
            metrics['DOR'][true_idx] = metrics['LR+'][true_idx]/metrics['LR-'][true_idx] # Diagnostic odds ratio (DOR)

        evaluation = pd.DataFrame(columns=list(range(conf_matrix.shape[0])))
        for name, metric in metrics.items():
            metric_ = dict()
            for class_idx, metric_value in metric.items():
                metric_[class_idx] = [metric_value]
            metric_ = pd.DataFrame(metric_).rename(index={0:name})
            evaluation = evaluation.append(metric_)
        return evaluation
   
    def imputation(self):
        pass

    def imbalance(self):
        pass

    def reduction(self):
        pass

    def linearity(self):
        pass
