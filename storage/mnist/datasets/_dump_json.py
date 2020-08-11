import json
from collections import OrderedDict

obj = {'a' : [1,2,3],
       'b' : {'a':1, 'b':2, 'c':3},
       'c' : (1,2,3),
       'd' : 1,
       'e' : 1.1,
       'f' : 'string'}

json.dump(obj, open('./dataset.json', 'w'))
