## Crawling
### Format
```python
import requests
from bs4 import BeautifulSoup

headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get('naver.com', headers=headers)
soup = BeautifulSoup(response.text, 'lxml')
```


## Examples
```python
import requests
from bs4 import BeautifulSoup
import pandas as pd

ticker = "AAPL"

headers = {'User-Agent': 'Mozilla/5.0'}
response = requests.get(f'https://finviz.com/quote.ashx?t={ticker}', headers=headers)

soup = BeautifulSoup(response.text, 'lxml') #html_tables = soup.find_all('table')
snapshot_table2 = soup.find('table', attrs={'class': 'snapshot-table2'})

finviz_tables = pd.read_html(str(snapshot_table2))
finviz_table = finviz_tables[0]
finviz_table.columns = ['key', 'value'] * 6
df_factor = pd.concat([finviz_table.iloc[:, i*2: i*2+2] for i in range(6)], ignore_index=True)
df_factor = df_factor.set_index('key').rename(columns={'value':ticker})
df_factor
```



