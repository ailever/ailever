from ..__base_structures import BaseNomenclature

import datetime
from pytz import timezone
import re

class MetadataStoreNomenclature(BaseNomenclature):
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        name = ''
        return name

    def search(self):
        pass

    def loading_connection(self):
        pass

    def storing_connection(self):
        pass
