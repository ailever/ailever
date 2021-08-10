import os


class ConceptualHierarchy:
    def __init__(self, name=None, level=None):
        self.__level = level if level else 0
        self.__name = name if name else None

    def __str__(self):
        return self.__name

    def hierarchy(self, name=None, level=None): 
        instance = super().__new__(type(self))
        instance.__init__(name=name)
        return instance

    def rename(self, name):
        self.__name = name

    @property
    def name(self):
        return self.__name

class BasePolicyHierarchy:
    def __init__(self, name=None, level=None):
        self.compile_first = True
        self.__path = ''
        self.__parent_name = ''
        self.__ascendants = list()
        self.__level = level if level else 0
        self.__name = name if name else None

    def __str__(self):
        return self.__name

    def hierarchy(self, name=None, level=None): 
        instance = super().__new__(type(self))
        instance.__init__(name=name)
        return instance

    def compiling(self, mkdir=False):
        if self.compile_first:
            if mkdir:
                path = str(self)
                if not os.path.isdir(path):
                    os.mkdir(path)

            children = self._compiling(self, mkdir=mkdir)
            while children:
                _children = list()
                for child in children:
                    _children.extend(self._compiling(child, mkdir))
                children = _children
        else:
            if mkdir:
                self.__init__(name=self.name)
                path = str(self)
                if not os.path.isdir(path):
                    os.mkdir(path)

            children = self._compiling(self, mkdir=mkdir)
            while children:
                _children = list()
                for child in children:
                    child.__init__(name=child.name)
                    _children.extend(self._compiling(child, mkdir))
                children = _children
        self.compile_first = False

    @staticmethod
    def _compiling(parent, mkdir=False):
        selected_objs = filter(lambda x: isinstance(x[1], type(parent)), vars(parent).items())
        children = list(map(lambda child: vars(parent)[child[0]], selected_objs))
        for child_obj in children:
            child_obj.parent_name = parent.name
            child_obj.ascendants = (parent.ascendants, parent.name)
            child_obj.path = child_obj.ascendants + [child_obj.name]
            if mkdir:
                path = child_obj.path
                if not os.path.isdir(path):
                    os.mkdir(path)
        return children

    @property
    def path(self):
        return self.__path

    @path.setter
    def path(self, bases):
        self.__path = os.path.join(*bases)

    @property
    def parent_name(self):
        return self.__parent_name

    @parent_name.setter
    def parent_name(self, name):
        self.__parent_name = name

    @property
    def ascendants(self):
        return self.__ascendants

    @ascendants.setter
    def ascendants(self, ascendants):
        self.__ascendants.extend(ascendants[0])
        self.__ascendants.append(ascendants[1])

    def rename(self, name):
        self.__name = name

    @property
    def name(self):
        return self.__name

    @property
    def _level(self):
        return self.__level


# Financial MLOps Basic Structure
fmlops_bs = ConceptualHierarchy('FMLOps_BasicStructure')
fmlops_bs.local_system = fmlops_bs.hierarchy('local_system')

fmlops = BasePolicyHierarchy('.fmlops')
fmlops_bs.local_system.root = fmlops
fmlops_bs.local_system.root.rawdata_repository = fmlops.hierarchy('rawdata_repository')
fmlops_bs.local_system.root.feature_store = fmlops.hierarchy('feature_store')
fmlops_bs.local_system.root.source_repository = fmlops.hierarchy('source_repository')
fmlops_bs.local_system.root.model_registry = fmlops.hierarchy('model_registry')
fmlops_bs.local_system.root.metadata_store = fmlops.hierarchy('metadata_store')
fmlops_bs.local_system.root.rawdata_repository.base_columns = ['date', 'close', 'volume']


def local_initialization_policy(local_environment:dict=None):
    global fmlops

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
    
    if local_environment:
        assert isinstance(local_environment, dict), 'The local_environment information must be supported by wtih dictionary data-type.'
        assert 'root' in local_environment.keys(), 'Set your root name.'
        assert 'rawdata_repository' in local_environment.keys(), 'Set your rawdata_repository name.'
        assert 'feature_store' in local_environment.keys(), 'Set your feature_store name.'
        assert 'source_repository' in local_environment.keys(), 'Set your source_repository name.'
        assert 'model_registry' in local_environment.keys(), 'Set your model_registry name.'
        assert 'metadata_store' in local_environment.keys(), 'Set your metadata_store name.'

        fmlops_bs.local_system.root.rename(local_environment['root'])
        fmlops_bs.local_system.root.rawdata_repository.rename(local_environment['rawdata_repository'])
        fmlops_bs.local_system.root.feature_store.rename(local_environment['feature_store'])
        fmlops_bs.local_system.root.source_repository.rename(local_environment['source_repository'])
        fmlops_bs.local_system.root.model_registry.rename(local_environment['model_registry'])
        fmlops_bs.local_system.root.metadata_store.rename(local_environment['metadata_store'])

    fmlops.compiling(mkdir=True)

    r"""
    - .fmlops
      |-- rawdata_repository
      |-- feature_store
      |-- source_repository
      |-- model_registry
      |-- metadata_store
    """
    


def remote_initialization_policy(remote_environment=None):
    pass

