from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
from sonar.models.blaser.loader import load_blaser_model
from .constants import sonar_lang_code_map


class MTQualityEstimator:
    def __init__(self):
        self.blaser_qe = load_blaser_model("blaser_2_0_qe").eval()
        print("Loaded blaser successfully")

        self.text_embedder = TextToEmbeddingModelPipeline(
            encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder",
        )
        print("Loaded text embedder successfully")
    
    def score(self, original_sentences, translated_sentences, source_lang, translated_lang):
        if type(original_sentences) == str:
            original_sentences = [original_sentences]
        if type(translated_sentences) == str:
            translated_sentences = [translated_sentences]

        src_embs = self.text_embedder.predict(original_sentences, source_lang=sonar_lang_code_map[source_lang])
        mt_embs = self.text_embedder.predict(translated_sentences, source_lang=sonar_lang_code_map[translated_lang])
        
        return self.blaser_qe(src=src_embs.mean(axis=0, keepdim=True), mt=mt_embs.mean(axis=0, keepdim=True)).item()  

  
def get_model():
    """ Returns MTQualityEstimator model. """
    return MTQualityEstimator()