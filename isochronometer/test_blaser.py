print("Running Blaser Test")
from . import blaser
if __name__ == "__main__":
    model = blaser.get_model()
    
    original_sentence = "this is a sentence."
    translated_sentence = "das ist ein satz."

    score = model.score(original_sentence, translated_sentence, source_lang="en", translated_lang="de")
    print(score)

    original_sentences = ["this is a sentence.", "this is another sentence."]   
    translated_sentences = ["das ist ein satz.", "das ist ein anderer satz."]

    accumulated_score = model.score(original_sentences, translated_sentences, source_lang="en", translated_lang="de")
    print(accumulated_score)