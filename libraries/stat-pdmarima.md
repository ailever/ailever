```python
import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm
from pmdarima import model_selection

# Data Loading and Split
data = pm.datasets.load_wineind()
train, test = model_selection.train_test_split(data, train_size=150)

# modeling
autoarima = pm.auto_arima(train,  
                          stationary=False,
                          with_intercept=True,
#                           start_p=0, d=None, start_q=0,
#                           max_p=5, max_d=1, max_q=5,
                          seasonal=True, m=12,
#                           start_P=0, D=None, start_Q=0,
#                           max_P=5, max_D=1, max_Q=5,
                          max_order=30, maxiter=5,
                          information_criterion='bic',
                          trace=True, suppress_warnings=True)
display(autoarima.summary())
pred_tr_ts_autoarima = autoarima.predict_in_sample()
pred_tr_ts_autoarima = autoarima.predict(n_periods=len(train))
pred_te_ts_autoarima = autoarima.predict(n_periods=len(test), 
                                         return_conf_int=True)[0]
pred_te_ts_autoarima_ci = autoarima.predict(n_periods=len(test), 
                                            return_conf_int=True)[1]

# visualization
ax = pd.DataFrame(test).plot(figsize=(12,4))
pd.DataFrame(pred_te_ts_autoarima, columns=['prediction']).plot(kind='line',
                                                                linewidth=3, fontsize=20, ax=ax)
ax.fill_between(pd.DataFrame(pred_te_ts_autoarima_ci).index,
                pd.DataFrame(pred_te_ts_autoarima_ci).iloc[:,0],
                pd.DataFrame(pred_te_ts_autoarima_ci).iloc[:,1], color='k', alpha=0.15)
plt.show()
```
