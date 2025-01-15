from beam import endpoint, Image

from transformers import (
    TokenClassificationPipeline,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from transformers.pipelines import AggregationStrategy
import numpy as np


# Define keyphrase extraction pipeline
class KeyphraseExtractionPipeline(TokenClassificationPipeline):
  def __init__(self, model, *args, **kwargs):
      super().__init__(
          model=AutoModelForTokenClassification.from_pretrained(model),
          tokenizer=AutoTokenizer.from_pretrained(model),
          *args,
          **kwargs
      )

  def postprocess(self, all_outputs):
      results = super().postprocess(
          all_outputs=all_outputs,
          aggregation_strategy=AggregationStrategy.FIRST,
      )
      return np.unique([result.get("word").strip() for result in results])
    

@endpoint(
    gpu="A100-40",
    name="inference",
    cpu=1,
    memory="1Gi",
    image=Image().add_python_packages(["numpy","transformers","torch"]),
)


def predict(**inputs):
    
    # Initialize the pipeline inside the function to ensure the environment is ready
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    emotion_pipeline = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", return_all_scores=False)

    # Load tokenizer and model
    # tokenizer = AutoTokenizer.from_pretrained("ml6team/keyphrase-extraction-distilbert-inspec")
    # model = AutoModelForTokenClassification.from_pretrained("ml6team/keyphrase-extraction-distilbert-inspec")

    keyword_pipeline = pipeline("token-classification", model="ml6team/keyphrase-extraction-distilbert-inspec")
    # keyword_pipeline = pipeline("token-classification", model=model, tokenizer=tokenizer)
    

    text = inputs.get("text", "No text provided")  # Default if no input is given
    emotions = emotion_pipeline(text)

    # keyphrases = keyword_pipeline(text)
    extractor = KeyphraseExtractionPipeline(model="ml6team/keyphrase-extraction-distilbert-inspec")
    keyphrases = extractor(text)

    # # Convert float32 scores to standard Python float
    # for keyphrase in keyphrases:
    #     keyphrase["score"] = float(keyphrase["score"])  # Convert to float (standard Python type)

    

    # Use the pipeline    
    # emotions = emotion_pipeline("I'm so happy...")

    return {"emotions": emotions, "keywords": list(keyphrases)}
