from abc import ABCMeta, abstractmethod


class BaseMemory(metaclass=ABCMeta):

    @abstractmethod
    def add(self):
        raise NotImplementedError

    @abstractmethod
    def sample(self):
        raise NotImplementedError
