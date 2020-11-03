## [Google Trend] | [pytrends](https://towardsdatascience.com/google-trends-api-for-python-a84bc25db88f) | [github](https://github.com/GeneralMills/pytrends)


### API
```python
import pandas as pd
from pytrends.request import TrendReq

# https://github.com/pat310/google-trends-api/wiki/Google-Trends-Categories
pytrend = TrendReq()
pytrend.categories()
```
`# trend build`
```python
pytrend = TrendReq()
pytrend.build_payload(kw_list=['Taylor Swift'], cat=0, timeframe='today 5-y', geo='', gprop='')
""" build_payload
timeframe is in ['all', 'now 1-H', 'now 4-H', 'now 1-d', 'now 7-d', 'today 1-m', 'today 3-m', 'today 12-m', 'today 5-y', '2016-12-14 2017-01-25', '2017-02-06T10 2017-02-12T07']
gprop is in ['', 'images', 'news', 'youtube', 'froogle']
"""
```
`# Interest by Region`
```python
pytrend = TrendReq()
pytrend.build_payload(kw_list=['Taylor Swift'])
pytrend.interest_by_region(resolution='COUNTRY', inc_low_vol=True, inc_geo_code=True)
""" interest_by_region
# resolution
'CITY' returns city level data
'COUNTRY' returns country level data
'DMA' returns Metro level data
'REGION' returns Region level data
"""
```
`# Interest Over Time`
```python
pytrend = TrendReq()
pytrend.build_payload(kw_list=['Taylor Swift'])
pytrend.interest_over_time()
```
`# Historical Hourly Interest`
```python
pytrend = TrendReq(proxies=['https://34.203.233.13:80'])
pytrend.get_historical_interest(keywords='korea', year_start=2020, month_start=1, day_start=1, hour_start=0, year_end=2020, month_end=2, day_end=1, hour_end=0, cat=0, geo='', gprop='', sleep=0)
```
`# Get Google Hot Trends data`
```python
pytrend = TrendReq()
df = pytrend.trending_searches(pn='united_states')
df.head()
```
`# Today Search`
```python
pytrend = TrendReq()
df = pytrend.today_searches(pn='US')
df.head()
```
`# Get Google Top Charts`
```python
pytrend = TrendReq()
df = pytrend.top_charts(2019, hl='en-US', tz=300, geo='GLOBAL')
df.head()
```
`# Get Google Keyword Suggestions`
```python
pytrend = TrendReq()
keywords = pytrend.suggestions(keyword='Mercedes Benz')
df = pd.DataFrame(keywords)
df.drop(columns= 'mid')   # This column makes no sense
```
`# Related Queries and Topic, returns a dictionary of dataframes`
```python
pytrend = TrendReq()
pytrend.build_payload(kw_list=['Coronavirus'])
related_queries = pytrend.related_queries()
related_topic = pytrend.related_topics()

rq = related_queries.values()
rt = related_topic.values()
rq = list(rq)[0]
rt = list(rt)[0]

rq_top = rq['top']
rq_rising = rq['rising']
rt_top = rt['top']
rt_rising = rt['rising']
```
