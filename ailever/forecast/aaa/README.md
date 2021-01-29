```python
from ailever.forecast.stock import krx

# download ver.
df = krx.all()
df = krx.kospi()
df = krx.kosdaq()
df = krx.konex()


# no-download ver.
df = krx._all()
df = krx._kospi()
df = krx._kosdaq()
df = krx._konex()


stocks = df[0]
columns = list(range(stocks.shape(1)))

info = df[1]
info.iloc[columns]

exception_list = df[2]
```
