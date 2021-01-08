from abc import ABCMeta, abstractmethod
 
class BaseAgent(metaclass=ABCMeta):
    @abstractmethod
    def __setup_policy(self):
        pass

    @abstractmethod
    def __setup_Q(self):
        pass

    @abstractmethod
    def micro_update_Q(self):
        pass
 
    @abstractmethod
    def macro_update_Q(self):
        pass

    @abstractmethod
    def judge(self):
        pass

