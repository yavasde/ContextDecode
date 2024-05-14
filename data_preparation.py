import spacy
from tqdm.auto import tqdm
import pandas as pd
from extract_word_sentences import extract_word_sentences


def get_features(sentences_as_features, feature_id=None):
    return list(set([sent[feature_id] for sent in sentences_as_features.values()]))

def get_feature_matrix(vocab, sentences_as_features):
    features = ["*sent_id*", "*sentence_str*", "*word_form*"]
    vocab.sort()
    features += vocab
    features += get_features(sentences_as_features, feature_id=1)
    features += get_features(sentences_as_features, feature_id=2)
    max_wordpos = max([s[3] for s in sentences_as_features.values()])
    features += [f"*p_{i}*" for i in range(max_wordpos + 1)]
    features += get_features(sentences_as_features, feature_id=4)
    features += get_features(sentences_as_features, feature_id=5)
    return features

def get_bag_of_words(sent_lemmas):
    return {lemma:1 for lemma in sent_lemmas}

def get_sentence_info(sent, word_lemma, vocab):
    lemmas = []
    doc = nlp(sent)
    for token in doc:
        if token.lemma_ == word_lemma:
            ngram_l = doc[token.i - 1].pos_
            ngram_r = doc[token.i + 1].pos_
            word_info = (
                f"*{token.tag_}*",
                f"*{token.dep_}*",
                token.i,
                f"*{ngram_l}_l*",
                f"*{ngram_r}_r*",
            )  # is upper?
        token_lemma = token.lemma_.lower()  # ?
        lemmas.append(token_lemma)
        if token_lemma not in vocab:
            vocab.append(token_lemma)
    return word_info, lemmas, vocab


target_word = "book" #any other word form? bought for buy try it!

nlp = spacy.load("en_core_web_trf")

print("Extracting Sentences")
sentences = extract_word_sentences(
    target_word, "brown", sentence_no=200
)  # select the sentence number based on your time/PC power


print("Annotating Features")
vocab = []
sentences_as_features = {}
progress_bar = tqdm(range(len(sentences)))
for i in range(len(sentences)):
    word_form = sentences[i][1]
    sent = sentences[i][0]
    word_info, lemmatized_sent, vocab = get_sentence_info(sent, target_word, vocab)
    sentences_as_features[i] = (
        lemmatized_sent,
        word_info[0],  # inflection
        word_info[1],  # deprel
        word_info[2],  # position
        word_info[3],  # 2gram-left
        word_info[4],  # 2gram-right
        sent,  # sentence
        word_form
    )
    progress_bar.update(1)

features = get_feature_matrix(vocab, sentences_as_features)
df = pd.DataFrame(columns=features)

# add sentences as features
for sent_no, sent_info in sentences_as_features.items():
    sent_features = {feat: 0 for feat in features}
    sent_dict = {
        "*sent_id*": sent_no,
        "*sentence_str*": sent_info[6],
        '*word_form*': sent_info[7],
        sent_info[1]: 1,
        sent_info[2]: 1,
        f"*p_{sent_info[3]}*": 1,
        sent_info[4]: 1,
        sent_info[5]: 1,
        
    }
    sent_features.update(sent_dict)
    sent_features.update(get_bag_of_words(sent_info[0]))
    df = df._append(sent_features, ignore_index = True)


df.to_pickle(f'data/{target_word}.pickle')