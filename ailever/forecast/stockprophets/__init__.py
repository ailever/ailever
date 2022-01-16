from .stockprophet_0000 import StockForecaster as SF0000


class StockProphet:
    def __init__(self, code, lag):
        self.MainForecaster = SF0000(code, lag)
        self.evaluation = self.MainForecaster.eval_table.copy()
    
    def forecast(self, model_name='GradientBoostingClassifier', trainstartdate='2015-03-01', teststartdate='2019-10-01', code=None, lag=None, comment=None):
        self.evaluation = self.MainForecaster.inference(model_name, transtartdate, teststartdate, lag, comment)
        self.dataset = self.MainForecaster.dataset.copy()
        self.model = self.MainForecaster.model
        return self.evaluation

    def analysis(self):
        pass

