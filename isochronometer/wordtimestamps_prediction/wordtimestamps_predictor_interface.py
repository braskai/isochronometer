from abc import ABC, abstractmethod
from typing import Optional

from .utils import Word

class WordTimestampsPredictor(ABC):
    @abstractmethod
    def predict(self, text: str, noise_scale: Optional[float], tokenized_words: Optional[list]) -> list[Word]:
        pass
