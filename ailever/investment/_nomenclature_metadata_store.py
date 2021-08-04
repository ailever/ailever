from abc import *

class MetadataStoreBaseNomenclature:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self):
        pass

class MetadataStoreNomenclature(MetadataStoreBaseNomenclature):
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        name = ''
        return name
