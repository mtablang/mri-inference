import logging
# from bs4 import BeautifulSoup
# import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import word_tokenize, sent_tokenize
# from nltk import pos_tag, ne_chunk, bigrams, trigrams
from collections import Counter
# import os



# load all the corpus from the nltk-data
# nltk_path = os.path.join(os.getcwd(), 'nltk-data')
# nltk.data.path.append(nltk_path)

# Utility function for named entity extraction
def extract_named_entities(tree):
    entities = []
    for subtree in tree:
        if hasattr(subtree, 'label') and subtree.label() == 'NE':
            entity = ' '.join(c[0] for c in subtree)
            entities.append(entity)
    return entities

# Get WordNet POS tag
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return nltk.corpus.wordnet.NOUN


def nlp_process(text):
    """
    Uses NLTK to extract the ngrams and named entities from the provided text.
    """

    logging.info("Extracting Keywords")

    # Tokenize text and get words (basically remove anything not alphanumeric)
    tokens = nltk.word_tokenize(text)
    words = [token for token in tokens if token.isalnum()]

    # Break up the text into sentences then tokenize the sentences and identify the parts of speech
    sentences = nltk.sent_tokenize(text)
    tokenized_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    pos_tagged_sentences = [nltk.pos_tag(sentence) for sentence in tokenized_sentences]

    # these are used in multiple places to look up stopwords
    stop_words = set(stopwords.words('english'))

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    # this will hold the lemmas with their surface words and occurrences initially but other keywords will be added
    keywords = {}
    # this var is for doing bigrams and trigrams, it is better to have lemmatized
    # terms rather than the surface words for SEO purposes. For now we'll just
    # throw out the surface words in this collection
    pos_tagged_sentences_lemmas = []
    for pos_sentence in pos_tagged_sentences:
        sentence_lemmas = []
        for pos_token in pos_sentence:
            token = pos_token[0].lower()
            if token.isalnum() and not token in stop_words:
                lemma = lemmatizer.lemmatize(token, pos=get_wordnet_pos(pos_token[1]))
                if lemma not in keywords:
                    keywords[lemma] = {
                        "surface_words": [], "occurrences": 0, "sources": ["Keyword"], "length": 1}
                if token not in keywords[lemma]["surface_words"]:
                    keywords[lemma]["surface_words"].append(token)
                keywords[lemma]["occurrences"] += 1
                sentence_lemmas.append([lemma, pos_token[1]])
            elif token.isalnum():
                sentence_lemmas.append([token, pos_token[1]])
        pos_tagged_sentences_lemmas.append(sentence_lemmas)

    # Find named entities
    chunked_ner_sentences = [nltk.ne_chunk(tagged_sentence, binary=True) for tagged_sentence in pos_tagged_sentences]
    entity_counts = Counter()
    for ne_tree in chunked_ner_sentences:
        entities = extract_named_entities(ne_tree)
        entity_counts.update(entities)
    entity_counts_dict = dict(entity_counts)
    # loop through the dict and lowercase the keys ensuring that if the key already exists in its
    # lowercase form, we combine the values rather than overwrite
    entities = {}
    for key, val in entity_counts_dict.items():
        lower_key = key.lower()
        if lower_key in entities:
            entities[lower_key] += val
        else:
            entities[lower_key] = val
    # loop through the entities now and combine them into the lemmas
    for key, val in entities.items():
        if key in keywords:
            # check to see who has the higher count and use that
            if val > keywords[key]["occurrences"]:
                keywords[key]["occurrences"] = val
            # add Named Entity to the sources
            keywords[key]["sources"].append("Named Entity")
        else:
            keywords[key] = {
                "surface_words": [key], "occurrences": val, "sources": ["Named Entity"], "length": len(key.split())}

    # Find collocations ...
    # Loop through the tagged & lemmatized sentences and extract the bigrams and trigrams that
    # meet the criteria of being either (Noun, Noun) or (Adjective, Noun) for bigram and
    # (Noun/Adjective, any, Noun/Adjective) for trigrams
    noun_or_adjective = ('JJ', 'JJR', 'JJS', 'NN', 'NNS', 'NNP', 'NNPS')
    noun_only = ('NN', 'NNS', 'NNP', 'NNPS')
    ngrams = {}
    for sentence in pos_tagged_sentences_lemmas:
        for bigram in nltk.bigrams(sentence):
            if bigram[0][1] in noun_or_adjective and bigram[1][1] in noun_only:
                # found a bigram, increment the counter
                bigram_text = bigram[0][0] + ' ' + bigram[1][0]
                if bigram_text not in ngrams:
                    # going to combine the surface words of the first word and the surface words of the second word
                    # into all possible permutations and store them as surface words for this bigram
                    surface_words_0 = [bigram[0][0]]
                    if bigram[0][0] in keywords:
                        surface_words_0 = keywords[bigram[0][0]]["surface_words"]
                    surface_words_1 = [bigram[1][0]]
                    if bigram[1][0] in keywords:
                        surface_words_1 = keywords[bigram[1][0]]["surface_words"]
                    surface_words = [f"{term1} {term2}" for term1 in surface_words_0 for term2 in surface_words_1]
                    ngrams[bigram_text] = {
                        "occurrences": 0, "surface_words": surface_words, "sources": ["Bigram"], "length": 2}
                ngrams[bigram_text]["occurrences"] += 1
        for trigram in nltk.trigrams(sentence):
            # in this situation we're going to look for the first and last words of the trigram to see if they
            # are an adjective or noun. If it holds true for both we'll add it to the list
            if trigram[0][1] in noun_or_adjective and trigram[2][1] in noun_or_adjective:
                trigram_text = trigram[0][0] + ' ' + trigram[1][0] + ' ' + trigram[2][0]
                if trigram_text not in ngrams:
                    # going to combine the surface words of all three words
                    # into all possible permutations and store them as surface words for this trigram
                    surface_words_0 = [trigram[0][0]]
                    if trigram[0][0] in keywords:
                        surface_words_0 = keywords[trigram[0][0]]["surface_words"]
                    surface_words_1 = [trigram[1][0]]
                    if trigram[1][0] in keywords:
                        surface_words_1 = keywords[trigram[1][0]]["surface_words"]
                    surface_words_2 = [trigram[2][0]]
                    if trigram[2][0] in keywords:
                        surface_words_2 = keywords[trigram[2][0]]["surface_words"]
                    surface_words = [f"{term1} {term2} {term3}" for term1 in surface_words_0 for term2 in surface_words_1 for term3 in surface_words_2]
                    ngrams[trigram_text] = {
                        "occurrences": 0, "surface_words": surface_words, "sources": ["Trigram"], "length": 3}
                ngrams[trigram_text]["occurrences"] += 1

    # OK we're going to combine the ngrams into the keywords
    for key, ngram in ngrams.items():
        if key in keywords:
            # already have one, let's combine the values together, making sure that whichever one has the highest count for
            # occurrences is used
            combined_sources = list(set(keywords[key]["sources"]).union(set(ngram["sources"])))
            combined_surface_words = list(set(keywords[key]["surface_words"]).union(set(ngram["surface_words"])))
            if ngram["occurrences"] > keywords[key]["occurrences"]:
                keywords[key]["occurrences"] = ngram["occurrences"]
            keywords[key]["sources"] = combined_sources
            keywords[key]["surface_words"] = combined_surface_words
        else:
            # fresh start, create a copy in the keywords
            keywords[key] = {ngram_key: ngram_value for ngram_key, ngram_value in ngram.items()}

    return {"keywords": keywords, "text": text}
