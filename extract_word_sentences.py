import nltk
from nltk.corpus import brown
import spacy
# from datasets import load_dataset
from tqdm.auto import tqdm


"""Extracts the sentences of a word from Brown corpus. Saves the sentences in a text file.
"""


def download_corpus(corpus_name):
    # if corpus_name == "bookcorpus":
    #     corpus = load_dataset("bookcorpus")
    #     corpus_sentences = [s['text'] for s in corpus['train']]
    # elif corpus_name == "wikipedia":
    #     corpus_sentences =  load_dataset("wikipedia").data['train']['text']
    # el
    if corpus_name == "brown":
        nltk.download("brown")
        corpus_sentences = brown.sents()
    # is there any other corpus to add?
    return corpus_sentences


def extract_word_sentences(target_word, corpus_name, sentence_no=200):
    """Extracts certain number of sentences of a word from Brown corpus.

    Returns:
    list: sentences (str) that contain the target word."""
    corpus = download_corpus(corpus_name)
    progress_bar = tqdm(range(sentence_no))
    sentences = []
    for sent in corpus:
        if target_word in sent:
            doc = nlp(" ".join(sent))
            for w in doc: # check this fast?
                if w.lemma_ == target_word:
                    word_form = w.text
                    sentences.append((doc.text, word_form))
                    progress_bar.update(1)
            if len(sentences) == sentence_no:
                return sentences
    return sentences



# different datasets?

nlp = spacy.load("en_core_web_sm")

