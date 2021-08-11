from ._base_policy import initialization_policy

local_environment = dict()
local_environment['root'] = '.analysis'
local_environment['exploratory_data_analysis'] = 'exploratory_data_analysis'
__analysis_bs__ = initialization_policy(local_environment)

from .ANALYSISInterfaces import EDA
from .ANALYSISInterfaces import counting
