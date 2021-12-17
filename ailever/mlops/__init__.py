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
        local_environment['root'] = input('[MLOps:Root]:')
        local_environment['feature_store'] = input('[MLOps:ROOT]:')
        local_environment['model_registry'] = input('[MLOps:ModelRegistry]:')
        local_environment['source_repository'] = input('[MLOps:SourceRepository]:')
        local_environment['metadata_store'] = input('[MLOps:MetadataStore]:')
    elif isinstance(local_environment, dict):
        keys = local_environment.keys()
        assert 'root' in keys, 'The local_environment must include the root.'
        assert 'feature_store' in keys, 'The local_environment must include path of the feature_store.'
        assert 'model_registry' in keys, 'The local_environment must include path of the model_registry.'
        assert 'source_repository' in keys, 'The local_environment must include path of the source_repository.'
        assert 'metadata_store' in keys, 'The local_environment must include path of the metadata_store.'
    else:
        local_environment = dict()
        local_environment['root'] = '.mlops'
        local_environment['feature_store'] = 'feature_store'
        local_environment['model_registry'] = 'model_registry'
        local_environment['source_repository'] = 'model_repository'
        local_environment['metadata_store'] = 'metadata_store'
    return initialization_policy(local_environment)

    

