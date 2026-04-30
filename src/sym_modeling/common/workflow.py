from abc import ABC, abstractmethod


class BaseTrainer(ABC):
    """Shared workflow interface for method-level trainers."""

    @abstractmethod
    def train(self):
        pass

    def predict(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate(self):
        pass
