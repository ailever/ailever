from ailever.investment import __fmlops_bs__ as fmlops_bs
from ..__base_structures import BaseManagement

import datetime
from pytz import timezone
import re

class ModelPredictionResultManager(BaseManagement):
    def __init__(self):
        self.__core = fmlops_bs.core['MPR']

    def __iter__(self):
        return self

    def __next__(self):
        name = self.__token['ticker']
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
        self.__token = specification
        specification['__loading_path_from_MPR__'] = self.__core.path
        specification['__loading_name_from_MPR__'] = None
        return specification

    def storing_connection(self, sepcification, usage='train'):
        self.__token = specification
        specification['__storing_path_in_MPR__'] = self.__core.path
        specification['__storing_name_in_MPR__'] = next(self)
        specification['__storing_process_regulation__'].append(
            ('MPR', 1),
        )
        return specification

