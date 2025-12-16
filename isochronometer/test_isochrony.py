from . import isochrony

if __name__ == "__main__":
    original_sentences = ["this is a sentence.", ""]   
    translated_sentences = ["das ist ein satz.", ""]
    original_start_times = ["00:00:40,025","00:00:46,167"]
    original_end_times = ["00:00:46,167","00:00:46,167"]


    ########################################################
    # PHASE 1 - Calculating metrics directly.
    print("Phase 1, end to end metrics")
    metric = isochrony.get_metric(source_lang="en", translated_lang="de")
    
    score = metric.evaluate(metric_type="ICM", original_sentences=original_sentences, translated_sentences=translated_sentences)
    print(f"ICM:{score}")
    
    score = metric.evaluate(metric_type="pred-SO", original_sentences=original_sentences, translated_sentences=translated_sentences)
    print(f"pred-SO:{score}")
    
    score = metric.evaluate( #SO requires start and end times.
        metric_type="SO", 
        original_sentences=original_sentences, 
        translated_sentences=translated_sentences, 
        original_start_times=original_start_times, 
        original_end_times=original_end_times
    )
    print(f"SO:{score}")


    ########################################################
    # PHASE 2
    print("Phase 2, if you want to use isochrony duration predictors on your own.")
    duration_predictor_en = isochrony.get_duration_predictor("en")
    duration_predictor_de = isochrony.get_duration_predictor("de")

    original_durations_actual, starts, ends = isochrony.calculate_actual_duration_start_end_from_time_lists(original_start_times, original_end_times)

    original_durations_predictions = duration_predictor_en.estimate(original_sentences)
    translated_duration_predictions = duration_predictor_de.estimate(translated_sentences)
    src_start, src_end = isochrony.calculate_start_end_lists_from_duration(original_durations_predictions)
    tgt_start, tgt_end = isochrony.calculate_start_end_lists_from_duration(translated_duration_predictions)
    print(src_start)
    print(src_end)
    print(tgt_start)
    print(tgt_end)

    speaker_rates = isochrony.calculate_speaker_rates(src_start, src_end, original_durations_predictions)
    print(speaker_rates)


    ########################################################
    # PHASE 3 - Additional Tests on additional langauges.
    print("Phase 3, testing exotic languages.")
    original_sentences = ["this is a sentence.", "that is another sentence"]   
    translated_sentences = ["das ist ein satz.", "das ist ein anderer satz"]
    translated_sentences_ar = ["هذا هو عبارة.", "هذا هو عبارة أخرى"]
    translated_sentences_ru = ["это предложение.", "это другое предложение"]
    translated_sentences_zh = ["这是句子。", "这是另一个句子"]
    translated_sentences_th = ["คำว่านี้คือข้อความ.", "คำว่านี้คือข้อความอื่น"]
    translated_sentences_ja = ["これは文です。", "これは別の文です"]
    translated_sentences_it = ["Questo è un esempio di testo italiano.", "Questo è un altro esempio di testo italiano"]
    translated_sentences_es = ["Este es un ejemplo de texto en español.", "Este es otro ejemplo de texto en español"]

    original_start_times = ["00:00:40,025","00:00:46,167"]
    original_end_times = ["00:00:46,167","00:00:46,167"]

    metric = isochrony.get_metric(source_lang="en", translated_lang="de")
    score = metric.evaluate(metric_type="pred-SO", original_sentences=original_sentences, translated_sentences=translated_sentences)
    print(f"pred-SO:{score}")

    metric = isochrony.get_metric(source_lang="en", translated_lang="ar")
    score = metric.evaluate(metric_type="pred-SO", original_sentences=original_sentences, translated_sentences=translated_sentences_ar)
    print(f"pred-SO:{score}")

    metric = isochrony.get_metric(source_lang="en", translated_lang="ru")
    score = metric.evaluate(metric_type="pred-SO", original_sentences=original_sentences, translated_sentences=translated_sentences_ru)
    print(f"pred-SO:{score}")

    metric = isochrony.get_metric(source_lang="en", translated_lang="zh")
    score = metric.evaluate(metric_type="pred-SO", original_sentences=original_sentences, translated_sentences=translated_sentences_zh)
    print(f"pred-SO:{score}")

    metric = isochrony.get_metric(source_lang="en", translated_lang="th")
    score = metric.evaluate(metric_type="pred-SO", original_sentences=original_sentences, translated_sentences=translated_sentences_th)
    print(f"pred-SO:{score}")

    # Japanese is not supported by Vits.
    # metric = isochrony.get_metric(source_lang="en", translated_lang="ja")
    # score = metric.evaluate(metric_type="pred-SO", original_sentences=original_sentences, translated_sentences=translated_sentences_ja)
    # print(f"pred-SO:{score}")

    # Italien is not supported by Vits.
    # metric = isochrony.get_metric(source_lang="en", translated_lang="it")
    # score = metric.evaluate(metric_type="pred-SO", original_sentences=original_sentences, translated_sentences=translated_sentences_it)
    # print(f"pred-SO:{score}")

    metric = isochrony.get_metric(source_lang="en", translated_lang="es")
    score = metric.evaluate(metric_type="pred-SO", original_sentences=original_sentences, translated_sentences=translated_sentences_es)
    print(f"pred-SO:{score}")
