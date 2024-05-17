from transformers import AutoTokenizer, AutoModel
from tqdm.auto import tqdm
import numpy as np


"""
Module for extracting sentence embeddings for a target word using a 
pre-trained language model. It utilizes the `transformers` library from 
Hugging Face to tokenize sentences and extract hidden state embeddings 
from a specified layer of the model.

Functions:

- extract_embeddings_target_word(df, layer_no=12): Extracts embeddings for 
    the target word in each sentence from a specified layer of the model.
"""


def extract_embeddings_target_word(df, layer_no=12):
    """
    Extracts embeddings for the target word in each sentence from a specified
    layer of the model.

    Parameters:
    df (DataFrame): DataFrame containing sentences and word forms.
    layer_no (int): The layer number from which to extract embeddings. Defaults to 12.

    Returns:
    np.array: Array of extracted embeddings.
    """
    print("Extracting the embeddings from the model")
    progress_bar = tqdm(range(len(df)))
    vectors = []
    for sent_id, d in df.iterrows():
        sentence = d["*sentence_str*"]
        word_form = d["*word_form*"]
        tokenized_sentence = tokenizer(sentence, truncation=True, return_tensors="pt")
        output = model(**tokenized_sentence, output_hidden_states=True).hidden_states
        word_inputid = tokenizer.convert_tokens_to_ids(word_form)
        if word_inputid != 100:
            token_index = (
                tokenized_sentence["input_ids"].tolist()[0].index(word_inputid)
            )
        vector = output[layer_no][:, token_index, :].detach().numpy()
        vectors.append(vector[0])
        progress_bar.update(1)
    return np.array(vectors)


model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
