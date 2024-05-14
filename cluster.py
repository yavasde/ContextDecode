from transformers import AutoTokenizer, AutoModel
import pandas as pd
from sklearn.cluster import KMeans
from tqdm.auto import tqdm


def extract_embeddings_target_word(sentence, target_word, layer_no=12):
    tokenized_sentence = tokenizer(sentence, truncation=True, return_tensors="pt")
    output = model(**tokenized_sentence, output_hidden_states=True).hidden_states
    word_inputid = tokenizer.convert_tokens_to_ids(target_word)
    if word_inputid != 100:
        token_id = tokenized_sentence["input_ids"].tolist()[0].index(word_inputid)
    vector = output[layer_no][:, token_id, :].detach().numpy()
    return vector


model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


target_word = "book"

df = pd.read_pickle(f"data/{target_word}.pickle")

vectors = []

print('Extracting embeddings from the model')
progress_bar = tqdm(range(len(df)))
for sent_id, d in df.iterrows():
    sentence = d["*sentence_str*"]
    word_form = d["*word_form*"]
    vector = extract_embeddings_target_word(sentence, word_form)
    vectors.append(vector[0])
    progress_bar.update(1)

print('Calculating the Ideal Cluster Number')
#Calculate the ideal cluster no

print('Clustering')
cluster_no = 3
model = KMeans(n_clusters=cluster_no, random_state=3, n_init=20, max_iter=1000)
clustering = model.fit(vectors)
cluster_labels = clustering.labels_
df.insert(2, "*cluster_labels*", cluster_labels, True)


df.to_pickle(f'data/{target_word}_clustered.pickle')

