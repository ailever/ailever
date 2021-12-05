from ailever.mlops import __mlops_bs__ as mlops_bs
from ..__base_structures import BaseManagement

import datetime
from pytz import timezone
import re

class MetadataStore(BaseManagement):
    def __init__(self):
        self.__core = mlops_bs.core['MS']

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

    def loading_connection(self, specification, usage='train'):
        specification['loading_path'] = self.__core.path
        specification['loading_name'] = None
        return specification

    def storing_connection(self, specification, usage='train'):
        specification['saving_path'] = self.__core.path
        specification['saving_name'] = None
        return specification



class MetadataStoreManager(BaseManagement):
    def __init__(self):
        self.__core = mlops_bs.core['MS']

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

    def loading_connection(self, specification, usage='train'):
        specification['loading_path'] = self.__core.path
        specification['loading_name'] = None
        return specification

    def storing_connection(self, specification, usage='train'):
        specification['saving_path'] = self.__core.path
        specification['saving_name'] = None
        return specification

