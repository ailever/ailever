```bash
$ pip install -U statsmodels
$ pip install monthdelta
$ pip install tabula-py
```

## Screener
```python
from ailever.investment import Screener

Screener.momentum_screener(baskets=['ARE', 'O', 'BXP'], period=10)
Screener.fundamentals_screener(baskets=['ARE', 'O', 'BXP'], sort_by='Marketcap')
Screener.pct_change_screener(baskets=['ARE', 'O', 'BXP'], sort_by=1
```

## Forecasting Model
```python
from ailever.investment import Forecaster

specifications = dict()
specifications['ARE'] = {'id':1, 'overwritten':True, 'architecture':'lstm00', 'framework':'torch', 'device':'cuda', 'batch_size':100, 'shuffle':False, 'drop_last':False, 'epochs':2, 'window':[5,10,20], 'base_columns':['date', 'close', 'volume'], 'packet_size':70, 'prediction_interval':30, 'start':'20180101', 'end':'20210816', 'rep':'ailever', 'message':'message', 'country':'united_states'}
specifications['BXP'] = {'id':2, 'overwritten':True, 'architecture':'lstm00', 'framework':'torch', 'device':'cuda', 'batch_size':100, 'shuffle':False, 'drop_last':False, 'epochs':2, 'window':[5,10,20], 'base_columns':['date', 'close', 'volume'], 'packet_size':70, 'prediction_interval':30, 'start':'20180101', 'end':'20210816', 'rep':'ailever', 'message':'message', 'country':'united_states'}
specifications['O'] = {'id':3, 'overwritten':True, 'architecture':'lstm00', 'framework':'torch', 'device':'cuda', 'batch_size':100, 'shuffle':False, 'drop_last':False, 'epochs':2, 'window':[5,10,20], 'base_columns':['date', 'close', 'volume'], 'packet_size':70, 'prediction_interval':30, 'start':'20180101', 'end':'20210816', 'rep':'ailever', 'message':'message', 'country':'united_states'}

forecaster = Forecaster()
forecaster.train_trigger(baskets=specifications.keys(), train_specifications=specifications)
forecaster.prediction_trigger(baskets=specifications.keys(), prediction_specifications=specifications)

#forecaster.forecasting_model_registry('remove')
#forecaster.forecasting_model_registry('clearall')
#forecaster.forecasting_model_registry('listdir')
forecaster.forecasting_model_registry('listfiles')
forecaster.model_prediction_result('listfiles')
```

## PortfolioManagement

```python
from ailever.investment import market_information, PortfolioManagement
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'NanumBarunGothic'

df = market_information()
pm = PortfolioManagement(baskets=df[df.Market=='KOSPI'].dropna().Symbol.to_list())
pm.portfolio_optimization(iteration=200)
```

```python
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = 'NanumBarunGothic'

Df = (pm.prllz_df[0][6500:], pm.prllz_df[1], pm.prllz_df[2], pm.prllz_df[3], pm.prllz_df[4])
pm.evaluate_momentum(Df, filter_period=30, regressor_criterion=0.8, capital_priority=False)
pm.portfolio_optimization(iteration=200)
```
