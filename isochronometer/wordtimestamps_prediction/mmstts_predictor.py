import math
import os

import langcodes as LC
import torch
import torch.nn as nn
import ttsmms.commons
import ttsmms.utils
from lhotse.workflows.forced_alignment.mms_aligner import _word_tokenize
from scipy.io.wavfile import write
from ttsmms import download
from ttsmms.models import (
    DurationPredictor,
    StochasticDurationPredictor,
    TextEncoder,
    commons,
)

from .utils import Word

from .mmstts_langcodes import mmstts_lang_to_model_map
from .vits_duration_predictor import VITSDurationPredictor
from .wordtimestamps_predictor_interface import WordTimestampsPredictor


class TextMapper(object):
    def __init__(self, vocab_file, lang):
        self.symbols = [x.replace("\n", "") for x in open(vocab_file, encoding="utf-8").readlines()]
        self.SPACE_ID = self.symbols.index(" ")
        self._symbol_to_id = {s: i for i, s in enumerate(self.symbols)}
        self._id_to_symbol = {i: s for i, s in enumerate(self.symbols)}
        self.lang = lang

    def text_to_sequence(self, text, cleaner_names):
        """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.

        Args:
        text: string to convert to a sequence
        cleaner_names: names of the cleaner functions to run the text through
        Returns:
        List of integers corresponding to the symbols in the text
        """
        sequence = []
        clean_text = text
        current_position = 0
        for symbol in clean_text:
            symbol_id = self._symbol_to_id[symbol]
            sequence += [symbol_id]
            current_position += len(symbol)
        return sequence

    def get_word_boundaries(self, text, is_add_blank):
        words = _word_tokenize(text)
        boundaries = []
        start = 0
        for word in words:
            end = start + len(word) * (2 if is_add_blank else 1)  # *2 because intersperse
            boundaries.append((start, end))
            start = end + (2 if is_add_blank else 1)  # +2 for the space character
        return words, boundaries

    def get_text(self, text, hps):
        words, word_boundaries = self.get_word_boundaries(text, hps.data.add_blank)
        len_before_filter = len(text)
        text = self.filter_oov(text)
        assert len_before_filter == len(text)
        text_norm = self.text_to_sequence(text.lower(), hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = ttsmms.commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm, words, word_boundaries

    def filter_oov(self, text):
        val_chars = self._symbol_to_id
        # Replace characters not in val_chars with a space
        text_filt = "".join([x if (x in val_chars or x.lower() in val_chars) else " " for x in text])
        return text_filt


class MmsTTSWordTimestampsPredictor(WordTimestampsPredictor):
    def __init__(self, lang: str, device: str = "cpu") -> None:
        lang = LC.get(lang).language  # leave only the top-level language

        if lang not in mmstts_lang_to_model_map:
            raise NotImplementedError(f"Language ", lang, " is not supported in mmstts")

        cache_dir = torch.hub.get_dir()
        model_path = download(mmstts_lang_to_model_map[lang], cache_dir)  # lang_code, dir for save model
        self.model_path = model_path
        self.vocab_file = f"{self.model_path}/vocab.txt"
        self.config_file = f"{self.model_path}/config.json"
        self.device = device
        assert os.path.isfile(self.config_file), f"{self.config_file} doesn't exist"
        self.hps = ttsmms.utils.get_hparams_from_file(self.config_file)
        self.text_mapper = TextMapper(self.vocab_file, lang)
        self.vits_dp = VITSDurationPredictor(
            len(self.text_mapper.symbols),
            **self.hps.model,
            use_sdp=True,
        )
        _ = self.vits_dp.eval().to(device)
        _ = ttsmms.utils.load_checkpoint(f"{self.model_path}/G_100000.pth", self.vits_dp, None)
        self.sampling_rate = self.hps.data.sampling_rate
        self.upsample_rate = int(math.prod(self.hps.model.upsample_rates))
        assert self.hps.data.training_files.split(".")[-1] != "uroman"

    def predict(self, text, noise_scale=0.2, tokenized_words: list = None) -> list[Word]:

        if len(text.strip()) == 0:
            return [Word(start=0, end=0.001, text=text)]

        stn_tst, words, word_boundaries = self.text_mapper.get_text(text, self.hps)
        assert len(words) == len(word_boundaries)
        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)])

            attn_mask = self.vits_dp.infer(
                x_tst.to(self.device), x_tst_lengths.to(self.device), noise_scale_w=noise_scale, length_scale=1.0
            )[0, 0]
            ts_words = []
            word_start = 0
            word_end = 0
            for word, boundary in list(zip(words, word_boundaries)):
                out_nonzero_positions = torch.nonzero(attn_mask[:, boundary[0] : boundary[1] + 1], as_tuple=True)[0]

                if boundary[0] != -1 and len(out_nonzero_positions) > 0:
                    out_nonzero_positions = out_nonzero_positions.detach().cpu()
                    word_start = self.upsample_rate / self.sampling_rate * out_nonzero_positions[0].item()
                    word_end = self.upsample_rate / self.sampling_rate * (1 + out_nonzero_positions[-1].item())
                else:  # removed word (all out-of-vocabulary symbols)
                    word_start = word_end
                    word_end = word_end + 0.001

                ts_words.append(
                    Word(
                        start=word_start,
                        end=word_end,
                        text=word,
                    )
                )
        return ts_words

    def to(self, device):
        self.vits_dp.to(device)
        self.device = device
        return self

    def parameters(self):
        return self.vits_dp.parameters()
