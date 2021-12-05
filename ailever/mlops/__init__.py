from ._base_policy import initialization_policy

local_environment = dict()
local_environment['root'] = '.mlops'
local_environment['feature_store'] = 'feature_store'
local_environment['model_registry'] = 'model_registry'
local_environment['source_repository'] = 'model_repository'
local_environment['metadata_store'] = 'metadata_store'
__mlops_bs__ = initialization_policy(local_environment)


def project(local_environment:dict=None):
    if local_environment is None:
        local_environment = dict()
        local_environment['root'] = '.mlops'
        local_environment['feature_store'] = 'feature_store'
        local_environment['model_registry'] = 'model_registry'
        local_environment['source_repository'] = 'model_repository'
        local_environment['metadata_store'] = 'metadata_store'
    else:
        assert isinstance(local_environment, dict), 'The local_environment must be an instance of dictionary.'
    return initialization_policy(local_environment)

    

