def dashboard(name, host='127.0.0.1', port='8050'):
    from ._dashboard import dashboard
    dashboard(name=name, host=host, port=port)

def Ailf_KR(Df=None, ADf=None, filter_period=300, capital_priority=True, regressor_criterion=2, seasonal_criterion=0.3, GC=False, V='KS11', download=False):
    from ._ailf_kr import Ailf_KR
    return Ailf_KR(Df=Df,
                   ADf=ADf,
                   filter_period=filter_period,
                   capital_priority=capital_priority,
                   regressor_criterion=regressor_criterion,
                   seasonal_criterion=seasonal_criterion,
                   GC=GC,
                   V=V,
                   download=download)
