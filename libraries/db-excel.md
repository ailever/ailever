
## Installation

```bash
$ pip install xlrd xlwt
```


### XLRD



### XLWT
```python
import xlwt
from ailever.dataset import UCI

book = xlwt.Workbook(encoding='utf-8')
sheet = book.add_sheet('sheet')
frame = UCI.adult(download=False)

for idx, row in frame.iterrows():
    sheet.write(*row)

book.save('result.xls')
```
