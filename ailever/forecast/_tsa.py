from .sarima import Process
from .hypothesis import ADFTest, LagCorrelationTest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from scipy import stats


class TSA:
    r"""
    Examples:
        >>> from ailever.forecast import TSA
        >>> ...
        >>> trendAR=[]; trendMA=[]
        >>> seasonAR=[]; seasonMA=[]
        >>> TSA.sarima((1,1,2), (2,0,1,4), trendAR=trendAR, trendMA=trendMA, seasonAR=seasonAR, seasonMA=seasonMA)
    """

    @classmethod
    def sarima(cls, trendparams:tuple=(0,0,0), seasonalparams:tuple=(0,0,0,1), trendAR=None, trendMA=None, seasonAR=None, seasonMA=None):
        Process(trendparams, seasonalparams, trendAR, trendMA, seasonAR, seasonMA)
 
    def __init__(self, TS, lag=1, title=None):
        self.models = dict()
        if not isinstance(TS, (pd.core.series.Series,)):
            self.TS = pd.Series(TS)
        else:
            self.TS = TS
        
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



    def analyze(self, TS, freq=None, lags=None, figsize=(18, 20), style='bmh'):
        if not isinstance(TS, pd.Series):
            TS = pd.Series(TS)

        with plt.style.context(style):
            fig = plt.figure(figsize=figsize)
            # mpl.rcParams['font.family'] = 'Ubuntu Mono'

            layout = (6, 2)
            ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
            dc_trend_ax = plt.subplot2grid(layout, (1, 0), colspan=2)
            dc_seasonal_ax = plt.subplot2grid(layout, (2, 0), colspan=2)
            dc_resid_ax = plt.subplot2grid(layout, (3, 0), colspan=2)
            acf_ax = plt.subplot2grid(layout, (4, 0))
            pacf_ax = plt.subplot2grid(layout, (4, 1))
            qq_ax = plt.subplot2grid(layout, (5, 0))
            pp_ax = plt.subplot2grid(layout, (5, 1))


            TS.plot(ax=ts_ax)
            ts_ax.set_title('Time Series')
            decompose = smt.seasonal_decompose(TS, model='additive', freq=freq)
            trend = decompose.trend
            trend.plot(ax=dc_trend_ax)
            dc_trend_ax.set_title('[Decompose] Time Series Trend')
            seasonal = decompose.seasonal
            seasonal.plot(ax=dc_seasonal_ax)
            dc_seasonal_ax.set_title('[Decompose] Time Series Seasonal')
            resid = decompose.resid
            resid.plot(ax=dc_resid_ax)
            dc_resid_ax.set_title('[Decompose] Time Series Resid')
            smt.graphics.plot_acf(TS, lags=lags, ax=acf_ax, alpha=0.5)
            smt.graphics.plot_pacf(TS, lags=lags, ax=pacf_ax, alpha=0.5)

            sm.qqplot(resid, line='s', ax=qq_ax)
            qq_ax.set_title('QQ Plot')
            stats.probplot(resid, sparams=(resid.mean(), resid.std()), plot=pp_ax)

            plt.tight_layout()
            plt.savefig('time_series_analysis.png')
            plt.show()

            return trend, seasonal, resid

