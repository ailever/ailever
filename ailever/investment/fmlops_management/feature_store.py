from ..__base_structures import BaseManagement

import datetime
from pytz import timezone
import re

class FeatureStoreManager(BaseManagement):
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        name = ''
        return name

    def _local_filesystem_user_interfaces(self):
        pass

    def _remote_filesystem_user_interfaces(self):
        pass

    def _local_search(self):
        pass

    def _remote_search(self):
        pass

    def local_loading_connection(self):
        pass

    def local_storing_connection(self):
        pass

    def remote_loading_connection(self):
        pass

    def remote_storing_connection(self):
        pass
