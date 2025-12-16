from isochronometer import blaser

# sample input
original_sentences = ["this is a sentence.", "this is another sentence."]   
translated_sentences = ["das ist ein satz.", "das ist ein anderer satz."]

# calculations
model = blaser.get_model()
blaser_score = model.score(original_sentences, translated_sentences, source_lang="en", translated_lang="de")
print(f"Blaser Score: {blaser_score}")