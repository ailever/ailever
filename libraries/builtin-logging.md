```python
import logging

logging.error("Message")
```


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

```python
import logging

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

handlers = dict()
handlers['stream'] = logging.StreamHandler()
handlers['stream'].setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
handlers['file'] = logging.FileHandler('test.log1')
handlers['file'].setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
logger.addHandler(handlers['stream'])
logger.addHandler(handlers['file'])

logger.critical('CRITICAL')
logger.error("ERROR")
logger.warning('WARNING')
logger.info("INFO")
logger.debug("DEBUG")
```
