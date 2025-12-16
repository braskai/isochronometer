import logging
from urllib.error import HTTPError

from .mmstts_predictor import (
    MmsTTSWordTimestampsPredictor,
)
from .wordtimestamps_predictor_interface import (
    WordTimestampsPredictor,
)

from .vits_chinese import BertChineseWordTimestampsPredictor


def build_timestamps_predictor(lang: str, device: str = "cpu") -> WordTimestampsPredictor:
    try:
        if lang.startswith("zh"):
            return BertChineseWordTimestampsPredictor(device=device)
        return MmsTTSWordTimestampsPredictor(lang=lang, device=device)
    except (HTTPError, NotImplementedError) as e:
        logging.warning(
            f"Error in build_timestamps_predictor: {lang}: {e}. Switching to simple syllables-based wordtimestamps predictor."
        )
        raise


if __name__ == "__main__":
    test_p = build_timestamps_predictor(lang="it")
    print(test_p.predict("Esempio di testo italiano"))
