from ..__base_structures import BaseNomenclature

class RawDataNomenclature(BaseNomenclature):
    def __init__(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        name = ''
        return name
