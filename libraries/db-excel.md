
## Installation

```bash
$ pip install xlrd xlwt
```


### XLRD
```python
import xlrd
import pandas as pd

book = xlrd.open_workbook("result.xls")
sheet = book.sheet_by_index(0)

print(book.sheet_names())
print(sheet.nrows, sheet.ncols)

rows = list()
for r_idx in range(sheet.nrows-1):
    rows.append(sheet.row_values(r_idx))

data = pd.DataFrame(rows)
data
```


### XLWT
```python
import xlwt
from ailever.dataset import UCI

book = xlwt.Workbook(encoding='utf-8')
sheet = book.add_sheet('sheet')
frame = UCI.adult(download=False)

for c_idx, c_name in enumerate(frame.columns):
    sheet.write(0, c_idx, c_name)
for r_idx, row in frame.iterrows():
    for c_idx, e in enumerate(row):
        sheet.write(r_idx+1, c_idx, e)

book.save('result.xls')
```
