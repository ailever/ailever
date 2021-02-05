from abc import ABCMeta, abstractmethod

class TypeCore:
    def __init__(self):
        self.__list = None
        self.__array = None
        self.__tensor = None
        self.__frame = None

    @property
    def list(self):
        return self.__list
    
    @list.setter
    def list(self, data):
        self.__list = data

    @property
    def array(self):
        return self.__array

    @array.setter
    def array(self, data):
        self.__array = data

    @property
    def tensor(self):
        return self.__tensor

    @tensor.setter
    def tensor(self, data):
        self.__tensor = data

    @property
    def frame(self):
        return self.__frame

    @frame.setter
    def frame(self, data):
        self.__frame = data
