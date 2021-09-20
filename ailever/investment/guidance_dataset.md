```python
from ailever.investment import market_information
df = market_information(baskets=None, only_symbol=False, inverse_mapping=False)
df
```
```python
from ailever.investment import market_information
df = market_information(baskets=['삼성전자', 'SK하이닉스'], only_symbol=True, inverse_mapping=False)
df
```

```python
from ailever.investment import market_information
df = market_information(baskets=['005930', '000660'], only_symbol=False, inverse_mapping=True)
df
```

```python
from ailever.investment import Loader
loader = Loader()
loader.into_local()
```

