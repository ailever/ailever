
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
for r_idx, row in frame.iterrows():
    for c_idx, e in enumerate(row):
        sheet.write(r_idx, c_idx, e)
book.save('result.xls')
```
