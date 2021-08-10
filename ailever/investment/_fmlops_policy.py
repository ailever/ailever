import os

r"""
from ._fmlops_policy import fmlops_bs

fmlops_bs.local_system

fmlops_bs.rawdata_repository
fmlops_bs.feature_store
fmlops_bs.model_registry
fmlops_bs.source_repository
fmlops_bs.metadata_store
"""

class PolicyHierarchy:
    def __init__(self, name=None):
        self.__name = name if name else None

    @property
    def name(self):
        return self.__name
        

fmlops_bs = PolicyHierarchy('FMLOps_Basic_Structure')
fmlops_bs.local_system = PolicyHierarchy('local_system')
fmlops_bs.rawdata_repository = PolicyHierarchy('rawdata_repository')
fmlops_bs.feature_store = PolicyHierarchy('feature_store')
fmlops_bs.model_registry = PolicyHierarchy('model_registry')
fmlops_bs.source_repository = PolicyHierarchy('source_repository')
fmlops_bs.metadata_store = PolicyHierarchy('metadata_store')


fmlops_bs.local_system.root = PolicyHierarchy('.fmlops')
fmlops_bs.local_system.rawdata_repository = PolicyHierarchy('rawdata_repository')
fmlops_bs.local_system.feature_store = PolicyHierarchy('feature_store')
fmlops_bs.local_system.source_repository = PolicyHierarchy('source_repository')
fmlops_bs.local_system.model_registry = PolicyHierarchy('model_registry')
fmlops_bs.local_system.metadata_store = PolicyHierarchy('metadata_store')

fmlops_bs.rawdata_repository.base_columns = ['date', 'close', 'volume']


def local_initialization_policy(local_environment:dict=None):
    r"""
    Usage:
        >>> # without arguments
        >>> from ._fmlops_policy import local_initialization_policy
        >>> local_initialization_policy()
       
        >>> # with arguments
        >>> from ._fmlops_policy import local_initialization_policy
        >>> local_environment = dict()
        >>> local_environment['rawdata_repository'] = 'rawdata_repository'
        >>> local_environment['feature_store'] = 'feature_store'
        >>> local_environment['source_repository'] = 'source_repository'
        >>> local_environment['model_registry'] = 'model_registry'
        >>> local_environment['metadata_store'] = 'metadata_store'
        >>> local_initialization_policy(local_environment=local_environment)
    """
    
    root = fmlops_bs.local_system.root.name
    if local_environment:
        assert isinstance(local_environment, dict), 'The local_environment information must be supported by wtih dictionary data-type.'
        assert 'rawdata_repository' in local_environment.keys(), 'Set your rawdata_repository path.'
        assert 'feature_store' in local_environment.keys(), 'Set your feature_store path.'
        assert 'source_repository' in local_environment.keys(), 'Set your source_repository path.'
        assert 'model_registry' in local_environment.keys(), 'Set your model_registry path.'
        assert 'metadata_store' in local_environment.keys(), 'Set your metadata_store path.'

        rawdata_repository = os.path.join(root, local_environment['rawdata_repository'])
        feature_store = os.path.join(root, local_environment['feature_store'])
        source_repository = os.path.join(root, local_environment['source_repository'])
        model_registry = os.path.join(root, local_environment['model_registry'])
        metadata_store = os.path.join(root, local_environment['metadata_store'])
    else:
        # Policy
        rawdata_repository = os.path.join(root, fmlops_bs.local_system.rawdata_repository.name)
        feature_store = os.path.join(root, fmlops_bs.local_system.feature_store.name)
        source_repository = os.path.join(root, fmlops_bs.local_system.source_repository.name)
        model_registry = os.path.join(root, fmlops_bs.local_system.model_registry.name)
        metadata_store = os.path.join(root, fmlops_bs.local_system.metadata_store.name)

    r"""
    - .fmlops
      |-- rawdata_repository
      |-- feature_store
      |-- source_repository
      |-- model_registry
      |-- metadata_store
    """
    
    if not os.path.isdir(root):
        os.mkdir(root)

    if not os.path.isdir(rawdata_repository):
        os.mkdir(rawdata_repository)
    
    if not os.path.isdir(feature_store):
        os.mkdir(feature_store)

    if not os.path.isdir(source_repository):
        os.mkdir(source_repository)

    if not os.path.isdir(model_registry):
        os.mkdir(model_registry)

    if not os.path.isdir(metadata_store):
        os.mkdir(metadata_store)


def remote_initialization_policy(remote_environment=None):
    pass

