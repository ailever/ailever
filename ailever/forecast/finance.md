## FINANCE

```python
import plotly.express as px
import FinanceDataReader as fdr

df = fdr.StockListing('KRX').set_index('Name')
stock_code = str(df[df.index == '삼성전자'].Symbol.values[0])
stock_price = fdr.DataReader(stock_code).reset_index()
px.line(stock_price, x='Date', y='Close')
```
![image](https://user-images.githubusercontent.com/52376448/98460226-ae464400-21e5-11eb-9de3-488e40dac1dc.png)
