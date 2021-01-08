from abc import ABCMeta, abstractmethod
 
class BaseAgent(metaclass=ABCMeta):
    @abstractmethod
    def micro_update_Q(self):
        pass
 
    @abstractmethod
    def macro_update_Q(self):
        pass

    @abstractmethod
    def judge(self):
        pass

