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

    def _local_filesystem_user_interfaces(self):
        pass

    def _remote_filesystem_user_interfaces(self):
        pass

    def _local_search(self):
        pass

    def _remote_search(self):
        pass

    def local_loading_connection(self, specification, usage='train'):
        specification['loading_path'] = self.__core.path
        specification['loading_name'] = None
        return specification

    def local_storing_connection(self, sepcification, usage='train'):
        specification['saving_path'] = self.__core.path
        specification['saving_name'] = None
        return specification

    def remote_loading_connection(self, sepcification, usage='train'):
        return specification

    def remote_storing_connection(self, sepcification, usage='train'):
        return specification
