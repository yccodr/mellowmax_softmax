from abc import ABC, abstractmethod


class Experiment(ABC):

    @abstractmethod
    def start(self):
        pass
