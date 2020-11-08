## FINANCE

```python
import plotly.express as px
import FinanceDataReader as fdr

df = fdr.StockListing('KRX').set_index('Name')
stock_code = str(df[df.index == '삼성전자'].Symbol.values[0])
stock_price = fdr.DataReader(stock_code, start='2020-07-01').rename_axis('price', axis=1)
scaled_price = (stock_price - stock_price.mean(axis=0))/stock_price.std()
```
```python
fig = px.line(scaled_price, x=scaled_price.index, y=scaled_price.columns, title='time series')
fig.update_xaxes(rangeslider_visible=True,
                 rangeselector=dict(buttons=list([dict(count=7, label="7d", step="day", stepmode="backward"),
                                                  dict(count=14, label="14d", step="day", stepmode="backward"),
                                                  dict(count=1, label="1m", step="month", stepmode="backward"),
                                                  dict(count=3, label="3m", step="month", stepmode="backward"),
                                                  dict(count=6, label="6m", step="month", stepmode="backward"),
                                                  dict(count=1, label="1y", step="year", stepmode="backward"),
                                                  dict(step="all")])))
fig.show()
```
![image](https://user-images.githubusercontent.com/52376448/98461032-6d056280-21ec-11eb-9e07-638c713ff5cc.png)

```python
fig = px.area(scaled_price, facet_col='price', facet_col_wrap=2)
fig.show()
```
![image](https://user-images.githubusercontent.com/52376448/98461057-860e1380-21ec-11eb-9a6d-176e3e5ad226.png)


```python
fig = px.scatter(stock_price, x=stock_price.index, y='Close', range_x=['2020-10-01', '2020-11-08'], title="Default Display with Gaps")
fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])]) #hide weekends
fig.show()
```
![image](https://user-images.githubusercontent.com/52376448/98461077-af2ea400-21ec-11eb-9373-7609a36f5102.png)

```python
fig = px.bar(scaled_price, x=scaled_price.index, y=scaled_price.columns)
fig.show()
```
![image](https://user-images.githubusercontent.com/52376448/98461090-c53c6480-21ec-11eb-9557-9a750889fbb0.png)

