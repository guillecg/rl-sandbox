from abc import ABCMeta, abstractmethod


class BaseAgent(metaclass=ABCMeta):

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def select_action(self):
        raise NotImplementedError
