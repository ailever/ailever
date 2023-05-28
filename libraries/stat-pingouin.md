- https://pingouin-stats.org/build/html/index.html

## ANOVA Dataset

```python
import pingouin as pg
import re

all_datasets = pg.list_dataset()
anova_datasets = list(filter(lambda x: re.search('anova', x.lower()), all_datasets.index.tolist()))
anova_datasets = all_datasets.loc[anova_datasets].copy()
display(anova_datasets)

pg.read_dataset('anova')
```
