from abc import ABCMeta, abstractmethod


class BaseNoise(metaclass=ABCMeta):

    @abstractmethod
    def sample(self):
        raise NotImplementedError

    @abstractmethod
    def reset(self):
        raise NotImplementedError
