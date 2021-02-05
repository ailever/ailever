from ._typecore_f import ForecastTypeCaster
from .sarima import Process
from .hypothesis import ADFTest, LagCorrelationTest
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.vector_ar.vecm import select_order, select_coint_rank
from scipy import stats


dummies = type('dummies', (dict,), {})
class DimensionError(Exception): pass
class TSA:
    r"""
    Examples:
        >>> from ailever.forecast import TSA
        >>> ...
        >>> trendAR=[]; trendMA=[]
        >>> seasonAR=[]; seasonMA=[]
        >>> TSA.sarima((1,1,2), (2,0,1,4), trendAR=trendAR, trendMA=trendMA, seasonAR=seasonAR, seasonMA=seasonMA)
    """

    @staticmethod
    def sarima(trendparams:tuple=(0,0,0), seasonalparams:tuple=(0,0,0,1), trendAR=None, trendMA=None, seasonAR=None, seasonMA=None):
        Process(trendparams, seasonalparams, trendAR, trendMA, seasonAR, seasonMA)
 
    def __init__(self, TS, lag=1, select_col=0, title=None):
        self.models = dict()
        self.dummies = dummies() 
        self.dummies.__init__ = dict()
        self.dummies.__init__['select_col'] = select_col

        TS = ForecastTypeCaster(TS, outtype='FTC')
        self._TS = TS

        # main univariate forecasting variable
        if TS.array.ndim == 1:
            self.TS = pd.Series(TS.array)
        elif TS.array.ndim == 2:
            self.TS = pd.Series(TS.array[:,select_col])
        else:
            raise DimensionError('TSA do not support dimension more than 2.')

        ADFTest(self.TS)
        LagCorrelationTest(self.TS, lag)
        with plt.style.context('ggplot'):
            plt.figure(figsize=(13,20))
            # mpl.rcParams['font.family'] = 'Ubuntu Mono'

            layout = (5, 2); axes = {}
            axes['0,0'] = plt.subplot2grid(layout, (0, 0), colspan=2)
            axes['1,0'] = plt.subplot2grid(layout, (1, 0))
            axes['1,1'] = plt.subplot2grid(layout, (1, 1))
            axes['2,0'] = plt.subplot2grid(layout, (2, 0), colspan=2)
            axes['3,0'] = plt.subplot2grid(layout, (3, 0), colspan=2)
            axes['4,0'] = plt.subplot2grid(layout, (4, 0))
            axes['4,1'] = plt.subplot2grid(layout, (4, 1))

            axes['0,0'].set_title('Time Series')
            axes['1,0'].set_title('Histogram')
            axes['1,1'].set_title('Lag Plot')
            axes['4,0'].set_title('QQ Plot')

            self.TS.plot(ax=axes['0,0'], marker='o')
            axes['1,0'].hist(self.TS)
            pd.plotting.lag_plot(self.TS, lag=lag, ax=axes['1,1'])
            smt.graphics.plot_acf(self.TS, ax=axes['2,0'], alpha=0.5)
            smt.graphics.plot_pacf(self.TS, ax=axes['3,0'], alpha=0.5)
            sm.qqplot(self.TS, line='s', ax=axes['4,0'])
            stats.probplot(self.TS, sparams=(self.TS.mean(), self.TS.std()), plot=axes['4,1'])

            plt.tight_layout()
            plt.show()


    def SARIMAX(self, exog=None, order=(2, 1, 0),
                seasonal_order=(0, 0, 0, 12), trend='ct',
                measurement_error=False, time_varying_regression=False,
                mle_regression=True, simple_differencing=False,
                enforce_stationarity=True, enforce_invertibility=True,
                hamilton_representation=False, concentrate_scale=False,
                trend_offset=1, use_exact_diffuse=False, dates=None,
                freq=None, missing='none', validate_specification=True,
                **kwargs):
        self.dummies.SARIMAX = dict()
        model = smt.SARIMAX(self.TS.values, exog=exog, order=order,
                            seasonal_order=seasonal_order, trend=trend,
                            measurement_error=measurement_error, time_varying_regression=time_varying_regression,
                            mle_regression=mle_regression, simple_differencing=simple_differencing,
                            enforce_stationarity=enforce_stationarity, enforce_invertibility=enforce_invertibility,
                            hamilton_representation=hamilton_representation, concentrate_scale=concentrate_scale,
                            trend_offset=trend_offset, use_exact_diffuse=use_exact_diffuse, dates=dates,
                            freq=freq, missing=missing, validate_specification=validate_specification,
                            **kwargs).fit()
        self.models['SARIMAX'] = model
        #self.models['SARIMAX'].test_serial_correlation(None)
        #self.models['SARIMAX'].test_heteroskedasticity(None)
        #self.models['SARIMAX'].test_normality(None)
        #self.models['SARIMAX'].arparams
        #self.models['SARIMAX'].maparams
        #self.models['SARIMAX'].seasonalarparams
        #self.models['SARIMAX'].seasonalmaparams
        #self.models['SARIMAX'].get_prediction(start=0, end=330).summary_frame(alpha=0.05)
        #self.models['SARIMAX'].aic
        #self.models['SARIMAX'].bic
        #self.models['SARIMAX'].mse

        return model.summary()

    def ETS(self, error="add", trend="add", damped_trend=True, seasonal="add", seasonal_periods=12,
            initialization_method="estimated", initial_level=None, initial_trend=None, initial_seasonal=None,
            bounds=None, dates=None, freq=None, missing="none"):
        model = smt.ETSModel(self.TS, error=error, trend=trend, damped_trend=damped_trend, seasonal=seasonal, seasonal_periods=seasonal_periods,
                             initialization_method=initialization_method, initial_level=initial_level, initial_trend=initial_trend, initial_seasonal=initial_seasonal,
                             bounds=bounds, dates=dates, freq=freq, missing=missing).fit(use_boxcox=True)
        self.dummies.ETS = dict()
        self.models['ETS'] = model
        #self.models['ETS'].test_serial_correlation(None)
        #self.models['ETS'].test_heteroskedasticity(None)
        #self.models['ETS'].test_normality(None)
        #self.models['ETS'].states
        #self.models['ETS'].get_prediction(start=0, end=330).summary_frame(alpha=0.05)
        #self.models['ETS'].aic
        #self.models['ETS'].bic
        #self.models['ETS'].mse

        return model.summary()

    def VECM(self, exog=None, exog_coint=None, dates=None,
             freq=None, missing="none", k_ar_diff=1, coint_rank=1,
             deterministic="ci", seasons=4, first_season=0):
        self.dummies.VECM = dict()
        data = self._TS.array
        lag_order = select_order(data=data, maxlags=10, deterministic="ci", seasons=4)
        rank_test = select_coint_rank(endog=data, det_order=0, k_ar_diff=lag_order.aic, method="trace", signif=0.05)

        model = smt.VECM(endog=data, exog=exog, exog_coint=exog_coint, dates=dates,
                         freq=freq, missing=missing, k_ar_diff=lag_order.aic, coint_rank=rank_test.rank,
                         deterministic=deterministic, seasons=seasons, first_season=first_season).fit()
        self.models['VECM'] = model
        #self.models['VECM'].test_granger_causality(caused=self.dummies.__init__['select_col'], signif=0.05).signif
        #self.models['VECM'].test_granger_causality(caused=self.dummies.__init__['select_col'], signif=0.05).pvalue
        #self.models['VECM'].test_inst_causality(causing=self.dummies.__init__['select_col']).summary()
        #self.models['VECM'].test_normality().test_statistic
        #self.models['VECM'].test_normality().crit_value
        #self.models['VECM'].test_normality().pvalue
        #self.models['VECM'].test_normality().summary()
        #self.models['VECM'].test_whiteness(nlags=12, adjusted=True).test_statistic
        #self.models['VECM'].test_whiteness(nlags=12, adjusted=True).crit_value
        #self.models['VECM'].test_whiteness(nlags=12, adjusted=True).pvalue
        #self.models['VECM'].test_whiteness(nlags=12, adjusted=True).summary()
        #self.models['VECM'].irf(periods=30)
        #self.models['VECM'].var_rep
        #self.models['VECM'].ma_rep(maxn=2)

        return model.summary()

