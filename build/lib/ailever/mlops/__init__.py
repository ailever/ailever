from ._base_policy import initialization_policy

local_environment = dict()
local_environment['root'] = '.mlops'
local_environment['feature_store'] = 'feature_store'
local_environment['model_registry'] = 'model_registry'
local_environment['source_repository'] = 'model_repository'
local_environment['metadata_store'] = 'metadata_store'
__analysis_bs__ = initialization_policy(local_environment)
