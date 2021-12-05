```python
import logging

logger = logging.getLogger("logger")
logger.setLevel(logging.WARNING)    
logger.addHandler(logging.StreamHandler())

logger.critical('CRITICAL')
logger.error("ERROR")
logger.warning('WARNING')
logger.info("INFO")
logger.debug("DEBUG")
```

```python
import logging

logger = logging.getLogger("logger")
logger.setLevel(logging.WARNING)    
logger.addHandler(logging.FileHandler('test.log'))

logger.critical('CRITICAL')
logger.error("ERROR")
logger.warning('WARNING')
logger.info("INFO")
logger.debug("DEBUG")
```
