import os
import pickle
import random
from abc import ABC, abstractmethod
import warnings

from .wordtimestamps_prediction import build_timestamps_predictor

class SpeakingTimeEstimator(ABC):
    @abstractmethod
    def __init__(self, lang):
        pass

    @abstractmethod
    def estimate(self, text: str) -> float:
        pass


class VitsSpeakingTimeEstimator(SpeakingTimeEstimator):
    def __init__(self, lang, device="cpu", finetuned=False) -> None:
        # print(f"Running on {device}")
        self.lang = lang
        self.predictor = build_timestamps_predictor(lang, device)

    def estimate(self, text: str) -> float:
        res = self.predictor.predict(text)
        if len(res) == 0:
            return min(0.1, len(text) / 15.0)
        speaking_time = res[-1].end
        return speaking_time
