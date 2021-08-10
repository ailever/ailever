from ._fmlops_policy import local_initialization_policy

local_environment = dict()
local_environment['root'] = '.fmlops'
local_environment['rawdata_repository'] = 'rawdata_repository'
local_environment['feature_store'] = 'feature_store'
local_environment['source_repository'] = 'source_repository'
local_environment['model_registry'] = 'model_registry'
local_environment['metadata_store'] = 'metadata_store'
fmlops_bs = local_initialization_policy(local_environment)

from .INVESTMENTInterfaces import initialize
from .INVESTMENTInterfaces import parallelize
from .INVESTMENTInterfaces import portfolio_optimize
from .INVESTMENTInterfaces import Forecaster
from .INVESTMENTInterfaces import sectors
from .INVESTMENTInterfaces import Loader
from .INVESTMENTInterfaces import Preprocessor
from .INVESTMENTInterfaces import screener




