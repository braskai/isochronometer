import warnings
from datetime import datetime
from tqdm import tqdm
import numpy as np

from .isochrony_utils import (
    VitsSpeakingTimeEstimator,
    SpeakingTimeEstimator,
)

from .constants import (
    ELEVENLABS_TO_VITS_DURATION_RATIOS,
    TTS_MAX_SCALE,
    TTS_MIN_SCALE,
)


class DurationPredictor:
    def __init__(self, lang):
        self.predictor = self.get_estimator(lang)

    def estimate(self, list_text):
        """Creates a duration estimate for each list member."""
        assert type(list_text) == list, "Duration Predictor accepts list of text only at this stage!"
        duration_prediction_list = []
        for text in list_text:
            duration = self.predict_word_timestamps_chunkwise(text)
            duration_prediction_list.append(duration)
        return duration_prediction_list

    # Run duration predictor by chunks (for long texts)
    def predict_word_timestamps_chunkwise(self, text, max_chunk_size=500):
        if len(text) <= 1:
            return 0.05
        chunks = self.chunk_text(text, chunk_size=max_chunk_size)
        predictions = [self.predictor.estimate(chunk) for chunk in chunks]
        return np.sum(predictions) 

    @staticmethod
    def chunk_text(text, chunk_size=500):
        words = text.split(" ")
        return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]

    @staticmethod
    def get_estimator(lang: str, model="vits", finetuned=False):
        if finetuned:
            warnings.warn("We don't have support for fine-tuned duration predictors yet.")
        if model == "vits":
            if "-" in lang:
                lang = lang.split("-")[0]
            if lang[:2] == "ja" or lang[:2] == "it":
                raise NotImplementedError(f"Does not work for {lang}!")
            # print("Creating Vits estimator")
            time_estimator = VitsSpeakingTimeEstimator(lang, finetuned=finetuned) #device="cuda"
        else:
            raise NotImplemented(f"We don't have model: {model} as duration predictor.")
        return time_estimator

def get_duration_predictor(lang):
    return DurationPredictor(lang)


def calculate_actual_duration_start_end_from_time_lists(start_time_list, end_time_list):
    """ Calculates the actual duration based on start_time and end_time """
    first_segment_time = None
    segment_duration_list = []
    start_list = []
    end_list = []
    for (start_index, start_time_string), (end_index, end_time_string) in tqdm(zip(enumerate(start_time_list), enumerate(end_time_list))):
        assert start_index == end_index
        time_format = "%H:%M:%S,%f"
        start_time = datetime.strptime(start_time_string, time_format)
        end_time = datetime.strptime(end_time_string, time_format)
        
        if first_segment_time is None:
            first_segment_time = start_time
        start_time = (start_time - first_segment_time).total_seconds()
        end_time = (end_time - first_segment_time).total_seconds()
        
        segment_duration_list.append(end_time-start_time)
        start_list.append(start_time)
        end_list.append(end_time)
    
    return segment_duration_list, start_list, end_list

def calculate_start_end_lists_from_duration(durations, offset_for_first_time=0.0):
    """ Calculates the start and end times based on segment wise durations. """
    first_segment_time = None
    start_list = []
    end_list = []
    current_start = offset_for_first_time

    for duration in durations:
        current_end = current_start + duration
        start_list.append(current_start)
        end_list.append(current_end)
        current_start = current_end
            
    return start_list, end_list

def calculate_speaker_rates(start_times, end_times, durations, min_val=0.82, max_val=1.4):
    """
    Given the same language we try to predict the speaker adapation rate w.r.t. to the metric.
    TODO:
    - We should adjust this to reflect speaker rates more accurately!
    """
    rates = []
    assert len(start_times) == len(end_times), "Start and End times have different lengths."
    assert len(start_times) == len(end_times), "Mismatch between duration length and segment lengths."

    for i in range(len(start_times)):
        rate = durations[i] / (0.1 + end_times[i] - start_times[i])
        rate = min(max_val, max(min_val, rate))
        rates.append(rate)

    return rates

def segment_iou(start1, end1, start2, end2):
    """
    Calculates intersection over union for a single segment.
    """
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    intersection_duration = max(0, overlap_end - overlap_start)

    union_start = min(start1, start2)
    union_end = max(end1, end2)
    union_duration = union_end - union_start

    if union_duration < 1e-3:
        return 1.0 if intersection_duration > 1e-3 else 0.0

    return intersection_duration / union_duration


def compute_iou(original_start, original_end, predicted_start, predicted_end, min_overlap_threshold=0.8):
    """Computes Speech Overlap (SO) metric"""
    assert len(original_start)==len(original_end), "Must be equal length"
    assert len(predicted_start)==len(predicted_end), "Must be equal length"
    assert len(original_start)==len(predicted_start), "Must be equal length"

    overlaps = []

    for idx in range(len(original_start)):
        overlap = segment_iou(original_start[idx], original_end[idx], predicted_start[idx], predicted_end[idx])
        overlaps.append(overlap)

    mean_overlap = sum(overlaps) / len(overlaps) if overlaps else 0
    # min_overlap = sum(1 for overlap in overlaps if overlap > min_overlap_threshold) / len(overlaps) if overlaps else 0

    return mean_overlap, np.min(overlaps) if overlaps else 0

def compute_icm(original_durations, translated_durations):
    """Computes the Classical ICM metric. abs(orig-trans)/orig"""
    assert len(original_durations) == len(translated_durations), "Must be same length."
    icms = []
    for idx in range(len(original_durations)):
        val = abs(original_durations[idx]-translated_durations[idx])/max(0.1,original_durations[idx])
        icms.append(val)
    
    mean_icm = sum(icms) / len(icms) if icms else 0 #0 indicates a perfect score!
    return mean_icm


def compute_penalities(rates):
    """Computes penalities (need to investigate meaning...)."""
    penalties = []
    for rate in rates:
        penalty = 0
        if rate < 0.95 or rate > 1.05:
            penalty = max(0, min(1, abs(rate - 1) - 0.05))
        penalties.append(penalty)
    mean_penalty = sum(penalties) / len(penalties) if penalties else 0
    return mean_penalty, penalties



class IsoChronyMetric:
    """Main Class for the IsoChronicMetric."""
    def __init__(self,source_lang, translated_lang):
        self.source_lang = source_lang
        self.translated_lang = translated_lang
        self.original_duration_predictor = get_duration_predictor(self.source_lang)
        self.translation_duration_predictor = get_duration_predictor(self.translated_lang)

    def evaluate(self, original_sentences, translated_sentences, original_start_times=None, original_end_times=None, metric_type="SO"):
        """
        Main function to calculate metrics. Some metrics require original start and end times as well.

        Metrics:
        - Speech Overlap (SO)
        - Min Speech Overlap (min-SO)
        - IsoChronoMeter (ICM) ["works fully on predicted values"]

        Future Metrics:
        - Speaker-rate Adjusted SO (SA-JO)
        - Speaker-rate Adjusted ICM (SA-ICM)
        ...
        """
        if metric_type=="SO":
            assert len(original_sentences) == len(translated_sentences), "Must be equal length."
            assert len(original_start_times) == len(original_end_times), "Must be equal length."
            assert len(original_sentences) == len(original_start_times), "Must be equal length."    
            predicted_durations = self.translation_duration_predictor.estimate(translated_sentences)
            _, orig_start, orig_end = calculate_actual_duration_start_end_from_time_lists(original_start_times,original_end_times)
            pred_start, pred_end = calculate_start_end_lists_from_duration(predicted_durations, orig_end[0])
            so, min_so = compute_iou(orig_start,orig_end,pred_start, pred_end)
            return so
        
        elif metric_type=="min-SO":
            assert len(original_sentences) == len(translated_sentences), "Must be equal length."
            assert len(original_start_times) == len(original_end_times), "Must be equal length."
            assert len(original_sentences) == len(original_start_times), "Must be equal length."    
            _, orig_start, orig_end = calculate_actual_duration_start_end_from_time_lists(original_start_times,original_end_times)
            pred_start, pred_end = calculate_start_end_lists_from_duration(predicted_durations, orig_end[0])
            so, min_so = compute_iou(orig_start,orig_end,pred_start, pred_end)
            return min_so
        
        elif metric_type=="pred-SO":  
            assert len(original_sentences) == len(translated_sentences), "Must be equal length."
            predicted_original_durations = self.original_duration_predictor.estimate(original_sentences)
            predicted_translated_durations = self.translation_duration_predictor.estimate(translated_sentences)
            original_pred_start, original_pred_end = calculate_start_end_lists_from_duration(predicted_original_durations)
            translated_pred_start, translated_pred_end = calculate_start_end_lists_from_duration(predicted_translated_durations)
            so, min_so = compute_iou(original_pred_start,original_pred_end,translated_pred_start, translated_pred_end)
            return so
        
        elif metric_type=="ICM":
            assert len(original_sentences) == len(translated_sentences), "Must be equal length."
            predicted_original_durations = self.translation_duration_predictor.estimate(translated_sentences)
            predicted_translated_durations = self.translation_duration_predictor.estimate(translated_sentences)
            icm = compute_icm(predicted_original_durations,predicted_translated_durations)
            return icm

        else:
            NotImplementedError(f"This metric is not implemented yet: {metric_type}")



def get_metric(source_lang, translated_lang):
    return IsoChronyMetric(source_lang, translated_lang)



# ######################################
# # TODO: Still to be integrated later. Below is the main metrics function from before.
# ######################################
#         lang_overlaps = []
#         lang_accuracies = []
#         cur_file_so = []
#         for speaker in pd.unique(df.speaker):
#             cur_speaker_df = df[df.speaker == speaker]
#             original_segments = cur_speaker_df[["original_start", "original_end"]].values
#             predicted_durations = cur_speaker_df["speaking_time"] * ELEVENLABS_TO_VITS_DURATION_RATIOS.get(lang_to, 1)
#             original_durations = cur_speaker_df.original_end - cur_speaker_df.original_start
#             original_predicted_durations = cur_speaker_df[
#                 "original_estimated_speaking_time"
#             ] * ELEVENLABS_TO_VITS_DURATION_RATIOS.get(lang_from, 1)
#             original_rolling_cumsum = original_durations.rolling(window=10, min_periods=1).sum()
#             correction_ratios = (
#                 original_rolling_cumsum / original_predicted_durations.rolling(window=10, min_periods=1).sum()
#             )
#             correction_ratios[original_rolling_cumsum < 3] = 1
#             correction_ratios[correction_ratios < (1.0 / TTS_MAX_SCALE)] = 1.0 / TTS_MAX_SCALE
#             correction_ratios[correction_ratios > (1.0 / TTS_MIN_SCALE)] = 1.0 / TTS_MIN_SCALE

#             speech_overlap, min_overlap, rates = compute_metrics(
#                 original_segments,
#                 (correction_ratios * predicted_durations).values,
#             )

#             speech_overlap_origin, _, _ = compute_metrics(original_segments, original_predicted_durations.values)
#             lang_overlaps.append(speech_overlap)
#             lang_accuracies.append(min_overlap)

#             cur_speaker_df.loc[:, "corrected_predicted_dur"] = correction_ratios * predicted_durations
#             cur_speaker_df.loc[:, "original_dur"] = original_durations
#             pred_dur_mismatch = abs(
#                 cur_speaker_df["original_estimated_speaking_time"] - cur_speaker_df["speaking_time"]
#             ) / (cur_speaker_df[["speaking_time", "original_estimated_speaking_time"]].max(axis=1) + 1e-3)
#             origin_pred_dur_mismatch = abs(
#                 cur_speaker_df["original_dur"] - cur_speaker_df["corrected_predicted_dur"]
#             ) / (cur_speaker_df[["original_dur", "corrected_predicted_dur"]].max(axis=1) + 1e-3)

#             if speech_overlap > -0.1 and speech_overlap <= 1.1:
#                 df_stat[lang_pair]["SO"].append(speech_overlap)
#                 cur_file_so.append(np.mean(pred_dur_mismatch))
#                 curoverlaps.append(speech_overlap)
#                 df_stat[lang_pair]["MinSO"].append(min_overlap)
#                 df_stat[lang_pair]["OriginVsPredAbsRelError"].append(np.mean(origin_pred_dur_mismatch))
#                 df_stat[lang_pair]["PredVsPredAbsRelError"].append(np.mean(pred_dur_mismatch))
#                 df_stat[lang_pair]["MaxRate"].append(np.max(rates))
#                 df_stat[lang_pair]["MinRate"].append(np.min(rates))
#                 df_stat[lang_pair]["SO_src"].append(speech_overlap_origin)
#         df_stat[lang_pair]["N"] += 1
#         all_files.append(f)
#         all_pids.append(f.split("/")[-1].split("__")[-1])
#         all_so.append(np.mean(cur_file_so))

#     for lang in df_stat:
#         df_stat[lang]["D"] = np.sum(df_stat[lang]["Duration"])
#         df_stat[lang]["SO"] = df_stat[lang]["SO"]
#         df_stat[lang]["MinSO"] = df_stat[lang]["MinSO"]
#         df_stat[lang]["SO_src"] = df_stat[lang]["SO_src"]
#     return df_stat, pd.DataFrame({"file": all_files, "so": all_so, "project_id": all_pids})