import spacy
from tqdm.auto import tqdm
import nltk
from nltk.corpus import brown
import pandas as pd


"""
Module for extracting sentences containing a target word from the Brown 
corpus and annotating them regarding their sentence context features. 
It includes methods to extract sentences from the Brown corpus, annotate sentences
with various linguistic features, and create a feature matrix for the whole dataset.

Functions:

- get_sentence_features(sent, word_lemma): Extracts linguistic features 
    from a sentence containing the target word.
- find_all_features(annotated_sentences, feature_index=None): Retrieves a 
    list of all unique features for a given feature type from the annotated 
    sentences.
- create_feature_list(annotated_sentences): Creates a list of features for 
    the feature matrix.
- extract_word_sentences(target_word, alternative_forms, n_sentences=200): 
    Extracts a specified number of sentences containing the target word from 
    the Brown corpus.
- annotate_sentences(sentences, target_word): Annotates sentences with various 
    linguistic features.
- create_feature_matrix(target_word, alternative_forms, n_sentences=200): 
    Creates a feature matrix for the target word based on extracted and annotated sentences.
"""


def get_sentence_features(sent, word_lemma):
    """
    Extracts linguistic features from a sentence containing the target word.

    Parameters:
    sent (str): Sentence containing the target word.
    word_lemma (str): Lemma of the target word.

    Returns:
    tuple: A tuple containing word information and a list of lemmas in the sentence.
    """
    sentence_lemmas = []
    doc = nlp(sent)
    for token in doc:
        if token.lemma_ == word_lemma:
            if token.i == 0:
                ngram_l = "firstword"
            else:
                ngram_l = doc[token.i - 1].pos_
            if token.i == len(doc) - 1:
                ngram_r = "lastword"
            else:
                ngram_r = doc[token.i + 1].pos_

            word_info = (
                f"*{token.tag_}*",
                f"*{token.dep_}*",
                token.i,
                f"*{ngram_l}_l*",
                f"*{ngram_r}_r*",
            )

        sentence_lemmas.append(token.lemma_.lower())
    return word_info, sentence_lemmas


def find_all_features(annotated_sentences, feature_index=None):
    """
    Retrieves a list of all unique features for a given feature type from the annotated sentences.

    Parameters:
    annotated_sentences (dict): Dictionary of annotated sentences.
    feature_index (int, optional): Index of the feature to retrieve. Defaults to None.

    Returns:
    list: List of unique features for the given feature type.
    """
    if feature_index == 0:
        vocab = []
        for word_info in annotated_sentences.values():
            for lemma in word_info[0]:
                if lemma not in vocab:
                    vocab.append(lemma)
        vocab.sort()
        return vocab
    else:
        return list(
            set(
                [
                    sentence_features[feature_index]
                    for sentence_features in annotated_sentences.values()
                ]
            )
        )


def create_feature_list(annotated_sentences):
    """
    Creates a list of features for the feature matrix.

    Parameters:
    annotated_sentences (dict): Dictionary of annotated sentences.

    Returns:
    list: List of features for the feature matrix.
    """
    feature_list = ["*sent_id*", "*sentence_str*", "*word_form*"]
    feature_list += find_all_features(annotated_sentences, feature_index=0)
    feature_list += find_all_features(annotated_sentences, feature_index=1)
    feature_list += find_all_features(annotated_sentences, feature_index=2)
    max_wordpos = max([s[3] for s in annotated_sentences.values()])
    feature_list += [f"*p_{i}*" for i in range(max_wordpos + 1)]
    feature_list += find_all_features(annotated_sentences, feature_index=4)
    feature_list += find_all_features(annotated_sentences, feature_index=5)
    return feature_list


def extract_word_sentences(target_word, alternative_forms, n_sentences=200):
    """
    Extracts a specified number of sentences containing the target word from the Brown corpus.

    Parameters:
    target_word (str): Target word to find in sentences.
    alternative_forms (str): Alternative forms of the target word, separated by backslashes.
    n_sentences (int): Number of sentences to extract. Defaults to 200.

    Returns:
    list: List of sentences (str) that contain the target word.
    """
    nltk.download("brown")
    corpus = brown.sents()
    progress_bar = tqdm(range(n_sentences))
    sentences = []
    for sentence in corpus:
        for word_form in alternative_forms:
            if " ".join(sentence) not in [
                sentence_info[0] for sentence_info in sentences
            ]:
                if word_form in sentence:
                    doc = nlp(" ".join(sentence))
                    for w in doc:
                        if w.lemma_ == target_word:
                            word_form = w.text
                            sentences.append((" ".join(sentence), word_form))
                            progress_bar.update(1)
                            break
        if len(sentences) == n_sentences:
            return sentences
    print(f"\nOnly {len(sentences)} sentences are found.")
    if len(sentences) < 30:
        raise Exception(
            "The number of sentence is too low. Try another word or another corpus."
        )
    return sentences


def annotate_sentences(sentences, target_word):
    """
    Annotates sentences with various linguistic features.

    Parameters:
    sentences (list): List of sentences containing the target word.
    target_word (str): Lemma of the target word.

    Returns:
    dict: Dictionary of annotated sentences.
    """
    annotated_sentences = {}
    progress_bar = tqdm(range(len(sentences)))
    for i in range(len(sentences)):
        word_form = sentences[i][1]
        sentence_str = sentences[i][0]
        word_info, lemmatized_sentence = get_sentence_features(
            sentence_str, target_word
        )
        annotated_sentences[i] = (
            lemmatized_sentence,  # for bag-of-words representation
            word_info[0],  # inflection
            word_info[1],  # deprel
            word_info[2],  # position
            word_info[3],  # 2gram-left
            word_info[4],  # 2gram-right
            sentence_str,  # sentence as string
            word_form,
        )
        progress_bar.update(1)
    return annotated_sentences


def create_feature_matrix(target_word, alternative_forms, n_sentences=200):
    """
    Creates a feature matrix for the target word based on extracted and annotated sentences.

    Parameters:
    target_word (str): Target word.
    alternative_forms (str): Alternative forms of the target word, separated by backslashes.
    n_sentences (int): Number of sentences to extract and annotate. Defaults to 200.

    Returns:
    DataFrame: DataFrame containing the feature matrix.
    """
    alternative_form_list = [form for form in alternative_forms.split("\\")]

    print(f"Extracting '{target_word}' Sentences")
    sentences = extract_word_sentences(
        target_word, alternative_form_list, n_sentences=n_sentences
    )

    print("Annotating Features")
    annotated_sentences = annotate_sentences(sentences, target_word)

    feature_list = create_feature_list(annotated_sentences)
    df = pd.DataFrame(columns=feature_list)

    for n_sentences, sentence_info in annotated_sentences.items():
        feature_dict = {feature: 0 for feature in feature_list}
        sentence_features = {
            "*sent_id*": n_sentences,
            "*sentence_str*": sentence_info[6],
            "*word_form*": sentence_info[7],
            sentence_info[1]: 1,
            sentence_info[2]: 1,
            f"*p_{sentence_info[3]}*": 1,
            sentence_info[4]: 1,
            sentence_info[5]: 1,
        }
        feature_dict.update(sentence_features)
        bag_of_words = {lemma: 1 for lemma in sentence_info[0]}
        feature_dict.update(bag_of_words)
        df = df._append(feature_dict, ignore_index=True)
    return df


nlp = spacy.load("en_core_web_sm")
