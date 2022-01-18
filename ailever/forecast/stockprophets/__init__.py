from .stockprophet_0000 import StockForecaster as SF0000


class StockProphet:
    def __init__(self, code, lag):
        self.MainForecaster = SF0000(code, lag)
        self.evaluation = self.MainForecaster.eval_table.copy()
        self.code = code
        self.lag = lag
    
        self.dataset = self.MainForecaster.dataset.copy()
        self.price = self.MainForecaster.price.copy()
        self.X = self.MainForecaster.X.copy()
        self.y = self.MainForecaster.y.copy()
        self.model = self.MainForecaster.model

    def forecast(self, model_name='GradientBoostingClassifier', trainstartdate='2015-03-01', teststartdate='2019-10-01', code=None, lag=None, comment=None, visual_on=True):
        self.evaluation = self.MainForecaster.inference(model_name, trainstartdate, teststartdate, code, lag, comment, visual_on)

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
        if lag is not None:
            self.lag = lag

        return self.evaluation

    def analysis(self):
        pass

