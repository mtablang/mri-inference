from beam import endpoint, Image, Volume

from transformers import TokenClassificationPipeline, AutoModelForTokenClassification, AutoTokenizer, pipeline
from transformers.pipelines import AggregationStrategy
import numpy as np

from custom_nlp import nlp_process

CACHE_PATH = "./weights"

def download_models():

    # Preload both models and tokenizers into the specified cache path
    keyphrase_model = AutoModelForTokenClassification.from_pretrained("ml6team/keyphrase-extraction-distilbert-inspec", cache_dir=CACHE_PATH)
    keyphrase_tokenizer = AutoTokenizer.from_pretrained("ml6team/keyphrase-extraction-distilbert-inspec", cache_dir=CACHE_PATH)
    emotion_pipeline = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", return_all_scores=False)

    return {
        "keyphrase_model": keyphrase_model,
        "keyphrase_tokenizer": keyphrase_tokenizer,
        "emotion_pipeline": emotion_pipeline,
    }

# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
    def __init__(self, model, tokenizer, *args, **kwargs):
        super().__init__(model=model, tokenizer=tokenizer, *args, **kwargs)

    def postprocess(self, all_outputs):
        results = super().postprocess(all_outputs=all_outputs, aggregation_strategy=AggregationStrategy.FIRST)
        return np.unique([result.get("word").strip() for result in results])

@endpoint(
    keep_warm_seconds=300,
    on_start=download_models,
    volumes=[Volume(name="weights", mount_path=CACHE_PATH)],
    gpu="A100-40",
    name="inference",
    cpu=1,
    memory="1Gi",
    image=Image().add_python_packages([
      "numpy",
      "transformers",
      "torch",
      "beautifulsoup4",
      "contractions",
      "nltk"
    ]),
)
def predict(context, **inputs):
    # Retrieve preloaded models and pipeline
    models = context.on_start_value
    keyphrase_model = models["keyphrase_model"]
    keyphrase_tokenizer = models["keyphrase_tokenizer"]
    emotion_pipeline = models["emotion_pipeline"]

    # Use the preloaded pipeline
    text = inputs.get("text", "No text provided")
    emotions = emotion_pipeline(text)

    # Use custom keyphrase extraction pipeline with preloaded model and tokenizer
    extractor = KeyphraseExtractionPipeline(model=keyphrase_model, tokenizer=keyphrase_tokenizer)
    keyphrases = extractor(text)

    if inputs.get("use_custom_nlp"):
      text = inputs.get("text", "No text provided")
      custom_result = nlp_process(text)
      return {"custom_result": custom_result}

    return {"emotions": emotions, "keywords": list(keyphrases)}