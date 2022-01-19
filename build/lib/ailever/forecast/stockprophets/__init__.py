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

    def simulate(self, model_name, code, max_lag, trainstartdate, invest_begin):
        results = list()
        for lag in range(1, max_lag):
            self.evaluation = self.MainForecaster.inference(model_name=model_name, trainstartdate=trainstartdate, teststartdate=invest_begin, code=code, lag=lag, comment=None, visual_on=False)

            self.dataset = self.MainForecaster.dataset.copy()
            self.price = self.MainForecaster.price.copy()
            self.X = self.MainForecaster.X.copy()
            self.y = self.MainForecaster.y.copy()
            self.model = self.MainForecaster.model

            if code is not None:
                self.code = code
            if lag is not None:
                self.lag = lag

            account = pd.DataFrame(data=np.c_[self.price.loc[invest_begin:].values.squeeze(), self.model.predict(self.X.loc[invest_begin:]).squeeze()], index=self.X.loc[invest_begin:].index.copy(), columns=['Price', 'Decision'])
            account['Lag'] = lag
            account = account.assign(Buy=lambda x: - x.Price * x.Decision)
            account = account.assign(Sell=lambda x: x.Price * (x.Decision*(-1)+1))
            account['Cash'] = account.assign(Cash=lambda x: x.Buy + x.Sell).Cash.cumsum() - account.Sell.astype(bool).sum()*account.Price[0]
            account['Cash'].iat[-1] = account['Cash'][-1] + account.Buy.astype(bool).sum()*account.Price[-1]
            
            invest = - account['Cash'][0]
            margin = account.Buy.astype(bool).sum()*account.Price[-1] - account.Sell.astype(bool).sum()*account.Price[0] + account.Sell.sum() + account.Buy.sum()
            profit = margin / invest
            invest_end = pd.DatetimeIndex([self.X.index[-1].strftime('%Y-%m-%d')], freq='B').shift(lag)[0].strftime('%Y-%m-%d')
            results.append([code, invest_begin, invest_end, lag, margin, invest, profit])
        report = pd.DataFrame(data=results, columns=['Code', 'Start', 'End', 'Lag', 'Margin', 'Invest', 'Profit'])
        self.account = account
        self.report = report
        return report

    def analysis(self):
        pass

