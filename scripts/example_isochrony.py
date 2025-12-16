from isochronometer import isochrony

# sample input
original_sentences = ["this is a sentence.", ""]   
translated_sentences = ["das ist ein satz.", ""]
original_start_times = ["00:00:40,025","00:00:46,167"]
original_end_times = ["00:00:46,167","00:00:46,167"]

# calculations
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