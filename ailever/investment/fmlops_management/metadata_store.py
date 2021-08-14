from ..__base_structures import BaseManagement

import datetime
from pytz import timezone
import re

class MetadataStoreManager(BaseManagement):
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        name = ''
        return name

    def _management(self):
        pass

    def _search(self):
        pass

    def loading_connection(self):
        pass

    def storing_connection(self):
        pass
