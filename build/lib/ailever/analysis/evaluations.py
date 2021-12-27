import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, auc

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
            # Positive
            metrics['PP'][pred_idx] = conf_matrix[:, pred_idx].sum() 
            # Negative
            metrics['PN'][pred_idx] = conf_matrix[:, :].sum() - conf_matrix[:, pred_idx].sum() 
            # hit
            metrics['TP'][pred_idx] = conf_matrix[pred_idx, pred_idx] 
            # type I error, false alarm, overestimation
            metrics['FP'][pred_idx] = conf_matrix[:, pred_idx].sum() - conf_matrix[pred_idx, pred_idx] 
            # correct rejection
            metrics['TN'][pred_idx] = np.trace(conf_matrix[:, :]) - conf_matrix[pred_idx, pred_idx] 
            # type II error, miss, underestimation
            metrics['FN'][pred_idx] = (conf_matrix[:, :].sum() - conf_matrix[:, pred_idx].sum()) - (np.trace(conf_matrix[:, :]) - conf_matrix[pred_idx, pred_idx]) 

        for true_idx in range(conf_matrix.shape[0]):
            pred_idx = true_idx
            # True positive rate (TPR), recall, sensitivity (SEN), probability of detection, hit rate, power
            metrics['TPR'][true_idx] = metrics['TP'][pred_idx]/metrics['P'][true_idx] if metrics['P'][true_idx] != 0 else np.inf
            # False positive rate (FPR), probability of false alarm, fall-out
            metrics['FPR'][true_idx] = metrics['FP'][pred_idx]/metrics['N'][true_idx] if metrics['N'][true_idx] != 0 else np.inf
            # False negative rate (FNR), miss rate
            metrics['FNR'][true_idx] = metrics['FN'][pred_idx]/metrics['P'][true_idx] if metrics['P'][true_idx] != 0 else np.inf
            # True negative rate (TNR), specificity (SPC), selectivity
            metrics['TNR'][true_idx] = metrics['TN'][pred_idx]/metrics['N'][true_idx] if metrics['N'][true_idx] != 0 else np.inf
            # Accuracy (ACC)
            metrics['ACC'][true_idx] = (metrics['TP'][pred_idx] + metrics['TN'][pred_idx])/(metrics['P'][true_idx] + metrics['N'][true_idx]) if metrics['P'][true_idx] + metrics['N'][true_idx] != 0 else np.inf
            # Balanced accuracy (BA)
            metrics['BA'][true_idx] = metrics['TPR'][true_idx]/metrics['TNR'][true_idx] if metrics['TNR'][true_idx] != 0 else np.inf
            # Informedness, bookmaker informedness (BM)
            metrics['BM'][true_idx] = metrics['TPR'][true_idx] + metrics['TNR'][true_idx] - 1 
            # Prevalence threshold (PT)
            metrics['PT'][true_idx] = (np.sqrt(metrics['TPR'][true_idx]*metrics['FPR'][true_idx])-metrics['FPR'][true_idx])/(metrics['TPR'][true_idx] - metrics['FPR'][true_idx]) if metrics['TPR'][true_idx] - metrics['FPR'][true_idx] != 0 else np.inf

        for pred_idx in range(conf_matrix.shape[1]):
            true_idx = pred_idx
            # Positive predictive value (PPV), precision
            metrics['PPV'][pred_idx] = metrics['TP'][pred_idx]/metrics['PP'][pred_idx] if metrics['PP'][pred_idx] != 0 else np.inf
            # False discovery rate (FDR)
            metrics['FDR'][pred_idx] = metrics['FP'][pred_idx]/metrics['PP'][pred_idx] if metrics['PP'][pred_idx] != 0 else np.inf
            # False omission rate (FOR)
            metrics['FOR'][pred_idx] = metrics['FN'][pred_idx]/metrics['PN'][pred_idx] if metrics['PN'][pred_idx] != 0 else np.inf
            # Negative predictive value (NPV)
            metrics['NPV'][pred_idx] = metrics['TN'][pred_idx]/metrics['PN'][pred_idx] if metrics['PN'][pred_idx] != 0 else np.inf
            # Markedness (MK), deltaP (Δp)
            metrics['MK'][pred_idx] = metrics['PPV'][pred_idx] + metrics['NPV'][pred_idx] - 1 
            # F1 score
            metrics['F1'][pred_idx] = (2*metrics['TP'][pred_idx])/(2*metrics['TP'][pred_idx] + metrics['FP'][pred_idx] + metrics['FN'][pred_idx]) if 2*metrics['TP'][pred_idx] + metrics['FP'][pred_idx] + metrics['FN'][pred_idx] != 0 else np.inf
            # Fowlkes–Mallows index (FM)
            metrics['FM'][pred_idx] = np.sqrt(metrics['PPV'][pred_idx]*metrics['TPR'][true_idx]) if not any([np.isinf(metrics['PPV'][pred_idx]), np.isinf(metrics['TPR'][true_idx])]) else np.nan
            # Matthews correlation coefficient (MCC)
            metrics['MCC'][pred_idx] = np.sqrt(metrics['TPR'][true_idx]*metrics['TNR'][true_idx]*metrics['PPV'][pred_idx]*metrics['NPV'][pred_idx]) - np.sqrt(metrics['FNR'][true_idx]*metrics['FPR'][true_idx]*metrics['FOR'][pred_idx]*metrics['FDR'][pred_idx]) if not any([np.isinf(metrics['TPR'][true_idx]), np.isinf(metrics['TNR'][true_idx]), np.isinf(metrics['PPV'][true_idx]), np.isinf(metrics['NPV'][true_idx]), np.isinf(metrics['FNR'][true_idx]), np.isinf(metrics['FPR'][true_idx]), np.isinf(metrics['FOR'][true_idx]), np.isinf(metrics['FDR'][true_idx])]) else np.nan
            # Threat score (TS), critical success index (CSI), Jaccard index
            metrics['JI'][pred_idx] = metrics['TP'][pred_idx] / (metrics['TP'][pred_idx] + metrics['FN'][pred_idx] + metrics['FP'][pred_idx]) if metrics['TP'][pred_idx] + metrics['FN'][pred_idx] + metrics['FP'][pred_idx] != 0 else np.inf

        for true_idx in range(conf_matrix.shape[0]):
            # Positive likelihood ratio (LR+)
            metrics['LR+'][true_idx] = metrics['TPR'][true_idx]/metrics['FPR'][true_idx] if metrics['FPR'][true_idx] != 0 else np.inf
            # Negative likelihood ratio (LR−)
            metrics['LR-'][true_idx] = metrics['FNR'][true_idx]/metrics['TNR'][true_idx] if metrics['TNR'][true_idx] != 0 else np.inf
            # Diagnostic odds ratio (DOR)
            metrics['DOR'][true_idx] = metrics['LR+'][true_idx]/metrics['LR-'][true_idx] if metrics['LR-'][true_idx] != 0 else np.inf

        evaluation = pd.DataFrame(columns=list(range(conf_matrix.shape[0])))
        for name, metric in metrics.items():
            metric_ = dict()
            for class_idx, metric_value in metric.items():
                metric_[class_idx] = [metric_value]
            metric_ = pd.DataFrame(metric_).rename(index={0:name})
            evaluation = evaluation.append(metric_)
        return evaluation
 
    @staticmethod
    def roc_curve(y_true, y_prod):
        # y_preds[target_class][threshold] : y_pred with nd.array type
        _y_preds = dict() 
        for target_class in np.unique(y_true):
            _y_preds[target_class] = dict()
            
            thresholds = list()
            for threshold in np.linspace(0, 1, 11):
                thresholds.append(threshold)
                _y_preds[target_class][threshold] = np.where(y_prod[:, target_class]>threshold, 1, 0)

        # y_preds[target_class] : y_pred with pd.DataFrame type by thresholds
        y_preds = dict()
        for target_class, pred_by_class in _y_preds.items():
             y_preds[target_class] = pd.DataFrame(pred_by_class)

        # _FPR_TPRs[target_class][threshold] : fpr, tpr
        _FPR_TPRs = dict()
        for target_class, y_pred in y_preds.items():
            _FPR_TPRs[target_class] = dict()
            for threshold in thresholds:
                _FPR_TPRs[target_class][threshold] = Evaluation.target_class_evaluation(y_true, y_pred[threshold]).loc[['FPR', 'TPR']][1].values

        # FPR_TPRs[target_class] : fpr, tpr by threshold
        FPR_TPRs = dict()
        AUCs = dict()
        for target_class, fpr_tpr in _FPR_TPRs.items():
            roc_frame = pd.DataFrame(fpr_tpr)
            FPR_TPRs[target_class] = roc_frame.copy().rename(index={0:'FPR', 1:'TPR'})
            AUCs[target_class] = auc(roc_frame.loc[0].values, roc_frame.loc[1].values)
        return FPR_TPRs, AUCs

    def imputation(self):
        pass

    def imbalance(self):
        pass

    def reduction(self):
        pass

    def linearity(self):
        pass
