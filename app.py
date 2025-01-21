from beam import endpoint, Image, Volume

from transformers import TokenClassificationPipeline, AutoModelForTokenClassification, AutoTokenizer, pipeline
from transformers.pipelines import AggregationStrategy
import numpy as np
import os
import nltk

from custom_nlp import nlp_process

CACHE_PATH = "./weights"

def download_models():
    NLTK_DATA_PATH = os.path.join(CACHE_PATH, 'nltk-data')
    nltk.data.path.append(NLTK_DATA_PATH) # Set NLTK path
    os.makedirs(NLTK_DATA_PATH, exist_ok=True) # Ensure the directory exists

    # Check and download NLTK resources only if they are missing
    def download_if_missing(resource_name):
        try:
            nltk.data.find(resource_name)
            print(f"{resource_name} already available.")
        except LookupError:
            print(f"Downloading {resource_name}...")
            nltk.download(resource_name, download_dir=NLTK_DATA_PATH)

    # Use the helper function for all required resources
    download_if_missing('punkt')
    download_if_missing('averaged_perceptron_tagger')
    download_if_missing('maxent_ne_chunker')
    download_if_missing('words')
    download_if_missing('wordnet')
    download_if_missing('stopwords')

    # Preload both models and tokenizers into the specified cache path
    keyphrase_model = AutoModelForTokenClassification.from_pretrained("ml6team/keyphrase-extraction-distilbert-inspec", cache_dir=CACHE_PATH)
    keyphrase_tokenizer = AutoTokenizer.from_pretrained("ml6team/keyphrase-extraction-distilbert-inspec", cache_dir=CACHE_PATH)
    emotion_pipeline = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", return_all_scores=False, model_kwargs={"cache_dir": CACHE_PATH})

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