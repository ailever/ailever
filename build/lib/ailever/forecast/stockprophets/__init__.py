from ...logging_system import logger
from .stockprophet_0000 import StockForecaster as SF0000


import graphviz
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz


class StockProphet:
    def __init__(self, code, lag_shift, sequence_length=5, trainstartdate=None, teststartdate=None):
        self.MainForecaster = SF0000(code, lag_shift, sequence_length, trainstartdate, teststartdate)
        self.evaluation = self.MainForecaster.eval_table.copy()
        self.code = code
        self.lag_shift = lag_shift
        self.sequence_length = sequence_length
    
        self.dataset = self.MainForecaster.dataset.copy()
        self.price = self.MainForecaster.price.copy()
        self.X = self.MainForecaster.X.copy()
        self.y = self.MainForecaster.y.copy()
        self.model = self.MainForecaster.model

    def evaluate(self, model_name='GradientBoostingClassifier', trainstartdate='2015-03-01', teststartdate='2019-10-01', code=None, lag_shift=None, sequence_length=None, comment=None, visual_on=True):
        if sequence_length is None:
            sequence_length = self.sequence_length
        self.evaluation = self.MainForecaster.validate(model_name, trainstartdate, teststartdate, code, lag_shift, sequence_length, comment, visual_on)

        """
        After feature selection, dataset is divided into X, y
        dataset : num(columns of dataset) > num(columns of X) + num(columns of y)
        """
        self.dataset = self.MainForecaster.dataset.copy()
        self.price = self.MainForecaster.price.copy()
        self.X = self.MainForecaster.X.copy()
        self.y = self.MainForecaster.y.copy()
        self.model = self.MainForecaster.model

        if code is not None:
            self.code = code
        if lag_shift is not None:
            self.lag_shift = lag_shift
        if sequence_length is not None:
            self.sequence_length = sequence_length

        return self.evaluation

    def simulate(self, model_name, code, min_lag, max_lag, sequence_length, trainstartdate, invest_begin):
        if sequence_length is None:
            sequence_length = self.sequence_length

        results = list()
        accounts = dict()
        for lag_shift in range(min_lag, max_lag):
            self.evaluation = self.MainForecaster.validate(model_name=model_name, trainstartdate=trainstartdate, teststartdate=invest_begin, code=code, lag_shift=lag_shift, sequence_length=sequence_length, comment='Simulation', visual_on=False)

            dataset = self.MainForecaster.dataset.copy()
            price = self.MainForecaster.price.copy()
            X = self.MainForecaster.X.copy()
            y = self.MainForecaster.y.copy()
            model = self.MainForecaster.model

            account = pd.DataFrame(data=np.c_[price.loc[invest_begin:].values.squeeze(), model.predict(X.loc[invest_begin:]).squeeze()], index=X.loc[invest_begin:].index.copy(), columns=['Price', 'Decision'])
            account['Code'] = code
            account['LagShift'] = lag_shift
            account['SequenceLength'] = sequence_length

            # investment strategy
            account = account.assign(Buy=lambda df: - df.Price * df.Decision)
            account = account.assign(Sell=lambda df: df.Price * (df.Decision*(-1)+1))
            account = account.assign(InitialShares=lambda df: (df.Sell.astype(bool)*(1)).sum())
            account = account.assign(CumulativeSharing=lambda df: (df.Buy.astype(bool)*(1) + df.Sell.astype(bool)*(-1)).cumsum())
            account = account.assign(FinalShares=lambda df: (df.Sell.astype(bool)*(1)).sum() + df.CumulativeSharing)
            account['Cash'] = account.assign(Cash=lambda df: df.Buy + df.Sell).Cash.cumsum() - account.Sell.astype(bool).sum()*account.Price[0]
            account['Cash'].iat[-1] = account['Cash'][-1] + account.Buy.astype(bool).sum()*account.Price[-1]

            invest = - account['Cash'][0]
            margin = account.Buy.astype(bool).sum()*account.Price[-1] - account.Sell.astype(bool).sum()*account.Price[0] + account.Sell.sum() + account.Buy.sum()
            profit = margin / invest
            invest_end = X.index[-1].strftime('%Y-%m-%d')
            results.append([code, invest_begin, invest_end, lag_shift, sequence_length, margin, invest, profit])
            accounts[lag_shift] = account[['Price', 'Decision', 'LagShift', 'SequenceLength', 'InitialShares', 'CumulativeSharing', 'FinalShares', 'Buy', 'Sell', 'Cash']].copy()
        report = pd.DataFrame(data=results, columns=['Code', 'Start', 'End', 'LagShift', 'SequenceLength', 'Margin', 'Invest', 'Profit'])
        self.accounts = accounts
        self.account = account
        self.report = report
        logger['forecast'].info('[Initail Investment Amount Validation Formula] Cash[0] = -InitialShares[0]*Price[0] + Buy[0] + Sell[0]')
        logger['forecast'].info('[Last Margin Validation Formula] Cash[-1] = Cash[-2] + Buy[-1] + Sell[-1] + FinalShares[-1]*Price[-1]')
        return report

    def forecast(self, model_name='GradientBoostingClassifier', comment=None, visual_on=True):
        pred = self.MainForecaster.inference(model_name, self.lag_shift, comment, visual_on)
        self.prediction = pred
        return pred

    def analyze(self, X, y, timeline, params={'max_depth':4, 'min_samples_split':100, 'min_samples_leaf':100}, plots={'FeatureImportance':True, 'DecisionTree':True, 'ClassificationReport':True}):

        if plots['Timeline']:
            import matplotlib.pyplot as plt
            timeline.plot(grid=True, figsize=(25,3))
            plt.show()

        model = getattr(self.MainForecaster, 'ModelDecisionTreeClassifier')(X.values, y.values.ravel(), params)
        model.fit(X, y)
        self._decision_tree_utils(model, X, y, plots)
        if plots['DecisionTree']:
            dot_data=export_graphviz(model, feature_names=X.columns, class_names=['decrease', 'increase'], filled=True, rounded=True)
            return graphviz.Source(dot_data)




    def _decision_tree_utils(self, model, X, y, plots):
        import matplotlib.pyplot as plt
        from sklearn import metrics

        # Feature importance
        if plots['FeatureImportance']:
            importance = model.feature_importances_
            plt.figure(figsize=(25,max(20, X.columns.shape[0]//4)))
            plt.barh([i for i in range(len(importance))], importance, tick_label=X.columns)
            plt.grid()
            plt.show()

        # Evaluation Metric
        if plots['ClassificationReport']:
            y_true = y 
            y_prob = model.predict_proba(X)
            y_pred = model.predict(X)

            confusion_matrix = metrics.confusion_matrix(y_true, y_pred)
            recall = confusion_matrix[1, 1]/(confusion_matrix[1, 0]+confusion_matrix[1, 1])
            fallout = confusion_matrix[0, 1]/(confusion_matrix[0, 0]+confusion_matrix[0, 1])
            precision = confusion_matrix[0, 0]/(confusion_matrix[0, 0]+confusion_matrix[1, 0])
            fpr, tpr1, thresholds1 = metrics.roc_curve(y_true, y_prob[:,1])
            ppv, tpr2, thresholds2 = metrics.precision_recall_curve(y_true, y_prob[:,1])

            print(metrics.classification_report(y_true, y_pred))
            print('- ROC AUC:', metrics.auc(fpr, tpr1))
            print('- PR AUC:', metrics.auc(tpr2, ppv))
            plt.figure(figsize=(25,7))
            ax1 = plt.subplot2grid((1,2), (0,0))
            ax2 = plt.subplot2grid((1,2), (0,1))
            ax1.plot(fpr, tpr1, 'o-') # X-axis(fpr): fall-out / y-axis(tpr): recall
            ax1.plot([fallout], [recall], 'bo', ms=10)
            ax1.plot([0, 1], [0, 1], 'k--')
            ax1.set_xlabel('Fall-Out')
            ax1.set_ylabel('Recall')
            ax2.plot(tpr2, ppv, 'o-') # X-axis(tpr): recall / y-axis(ppv): precision
            ax2.plot([recall], [precision], 'bo', ms=10)
            ax2.plot([0, 1], [1, 0], 'k--')
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            plt.show()

    def observe(self):
        return


