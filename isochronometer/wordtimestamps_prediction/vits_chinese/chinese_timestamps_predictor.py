import math

import numpy as np
import torch

from isochronometer.wordtimestamps_prediction.utils import Word
from isochronometer.wordtimestamps_prediction.vits_duration_predictor import (
    VITSDurationPredictor,
)
from isochronometer.wordtimestamps_prediction.wordtimestamps_predictor_interface import (
    WordTimestampsPredictor,
)
from ttsmms.utils import get_hparams_from_file

from .text import cleaned_text_to_sequence
from .text.symbols import symbols
from .vits_pinyin import VITS_PinYin

model_config = {
    "inter_channels": 192,
    "hidden_channels": 192,
    "filter_channels": 768,
    "n_heads": 2,
    "n_layers": 6,
    "kernel_size": 3,
    "p_dropout": 0.1,
    "resblock": "1",
    "resblock_kernel_sizes": [3, 7, 11],
    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
    "upsample_rates": [8, 8, 2, 2],
    "upsample_initial_channel": 512,
    "upsample_kernel_sizes": [16, 16, 4, 4],
    "n_layers_q": 3,
    "use_spectral_norm": False,
    "sampling_rate": 16000,
}


class BertChineseWordTimestampsPredictor(WordTimestampsPredictor):
    def __init__(
        self,
        bert_path="hfl/chinese-bert-wwm",
        device="cuda",
        max_text_size=500,
    ):
        self.device = device

        self.tts_front = VITS_PinYin(bert_path, device)
        self.net_g = VITSDurationPredictor(
            len(symbols),
            **model_config,
        ).to(device)
        state_dict = torch.hub.load_state_dict_from_url(
            "https://github.com/PlayVoice/vits_chinese/releases/download/v1.0/vits_bert_model.pth",
            map_location="cpu",
        )
        self.net_g.load_state_dict(state_dict["model"], strict=False)
        # _ = load_checkpoint(vits_path, self.net_g, None)
        self.upsample_rate = int(math.prod(model_config["upsample_rates"]))
        self.max_text_size = max_text_size  # Limited by max_size in BERT absolute position encoding

    def estimate_chunk(self, text: str) -> list[Word]:
        phonemes, char_embeds, chars, count_phones = self.tts_front.chinese_to_phonemes(text)
        if len(chars) == 0:
            return [Word(start=0, end=0.1, text=text)], 0.1
        chars = chars.replace("[PAD]", " ")
        input_ids = cleaned_text_to_sequence(phonemes)
        with torch.no_grad():
            x_tst = torch.LongTensor(input_ids).unsqueeze(0).to(self.device)
            x_tst_lengths = torch.LongTensor([len(input_ids)]).to(self.device)
            x_tst_prosody = torch.FloatTensor(char_embeds).unsqueeze(0).to(self.device)
            result = self.net_g.infer(x_tst, x_tst_lengths, x_tst_prosody, length_scale=1)[0, 0]

        start_index = 0
        current_time = 0.0
        time_per_frame = self.upsample_rate / model_config["sampling_rate"]
        ts_chars = []
        for char, count in zip(chars, count_phones):
            end_index = start_index + count

            out_nonzero_positions = torch.nonzero(result[:, start_index:end_index], as_tuple=True)[0]

            if start_index != -1 and len(out_nonzero_positions) > 0:
                out_nonzero_positions = out_nonzero_positions.detach().cpu()
                char_start = time_per_frame * out_nonzero_positions[0].item()
                char_end = time_per_frame * (1 + out_nonzero_positions[-1].item())
            else:  # removed character (all out-of-vocabulary symbols)
                char_start = current_time
                char_end = current_time + 0.001

            if char != " ":
                ts_chars.append(
                    Word(
                        start=char_start,
                        end=char_end,
                        text=char,
                    )
                )
            start_index = end_index
        return ts_chars, result.shape[0] * time_per_frame

    def predict(self, text: str, noise_scale=0.2, tokenized_words=None) -> list[Word]:
        if len(text) <= self.max_text_size:
            return self.estimate_chunk(text)[0]

        all_words = []
        total_duration = 0.0
        # Split the text into chunks
        chunks = [text[i : i + self.max_text_size] for i in range(0, len(text), self.max_text_size)]

        for chunk in chunks:
            chunk_words, chunk_duration = self.estimate_chunk(chunk)

            # Adjust timestamps for the current chunk
            for word in chunk_words:
                word.start += total_duration
                word.end += total_duration
                all_words.append(word)

            total_duration += chunk_duration

        return all_words

    def to(self, device):
        self.net_g.to(device)
        self.device = device
        return self

    def parameters(self):
        return self.net_g.parameters()


if __name__ == "__main__":
    mod = BertChineseWordTimestampsPredictor()
    speaking_time = mod.estimate("时光总是慢慢地偷走我们的容颜，渐渐地有些人终将离我们而远去")
    print(speaking_time)
