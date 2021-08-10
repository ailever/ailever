from ._fmlops_policy import local_initialization_policy
from ._fmlops_policy import fmlops_bs as __fmlops_bs__

local_environment = dict()
local_environment['root'] = __fmlops_bs__.local_system.root.name
local_environment['rawdata_repository'] = __fmlops_bs__.local_system.root.rawdata_repository.name
local_environment['feature_store'] = __fmlops_bs__.local_system.root.feature_store.name
local_environment['source_repository'] = __fmlops_bs__.local_system.root.source_repository.name
local_environment['model_registry'] = __fmlops_bs__.local_system.root.model_registry.name
local_environment['metadata_store'] = __fmlops_bs__.local_system.root.metadata_store.name
local_initialization_policy(local_environment)

from .INVESTMENTInterfaces import initialize
from .INVESTMENTInterfaces import parallelize
from .INVESTMENTInterfaces import portfolio_optimize
from .INVESTMENTInterfaces import Forecaster
from .INVESTMENTInterfaces import sectors
from .INVESTMENTInterfaces import Loader
from .INVESTMENTInterfaces import Preprocessor
from .INVESTMENTInterfaces import screener




