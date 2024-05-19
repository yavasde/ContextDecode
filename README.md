# ContextDecode: Reverse Engineering Contextualized Embedding Clusters for Automated Interpretation

ContextDecode is a tool designed to automatically interpret contextualized word embedding clusters of a word by identifying the sentence context features responsible for their formation.

## Method Details


<img src="https://github.com/yavasde/ContextDecode/assets/56029511/a6a8664d-4ee9-426d-bfa5-8c5b2e102f0f" width="500">

_Figure 1: Method_


For a detailed explanation of the method behind ContextDecode, please refer to the accompanying blog post [here](https://medium.com/@deniz.eyavas/contextdecode-reverse-engineering-for-automated-interpretation-of-contextualized-embedding-e27882275f82).



## Usage

This method has been successfully applied in the following publication:

Deniz Ekin Yavas. 2024. "Assessing the Significance of Encoded Information in Contextualized Representations to Word Sense Disambiguation." In Proceedings of the Third Workshop on Understanding Implicit and Underspecified Language, pages 42–53, Malta. Association for Computational Linguistics. [link](https://aclanthology.org/2024.unimplicit-1.4/)

## Installation

To use ContextDecode, follow these steps:

1. Clone the project repository.
2. Create a virtual environment for the project.
3. Download the required libraries using the provided requirements.txt file.

Once installed, you're ready to work with the code!

## Running ContextDecode

### Argument Definitions

- **target_word**: The word for which contextualized embedding clusters will be interpreted.
- **alternative_forms**: Alternative forms of the word, provided as a string with backslashes separating each form (e.g., 'buy\bought').
- **n_sentences**: Number of sentences to extract for analysis.
- **n_features**: Number of sentence features to select for interpretation.

### Example:

```bash
python main.py foot foot\feet 200 10
```

## Results Explanation

ContextDecode produces two main files as results:

1. A file containing example sentences with the important features annotated.
2. A t-SNE visualization of the clusters annotated with the important features.

<img src="https://github.com/yavasde/ContextDecode/assets/56029511/b61711f8-db16-4547-9f6b-c3010f40c33a" width="500">

_Figure 2: t-SNE visualization of “foot” instances, their clusters, and important features of each cluster_


Additionally, the performance of the classifier trained with the top-10 features is provided. This is compared to a baseline classifier trained with random 10 features, offering insights into the effectiveness of the selected features for interpreting the clusters.

If no features are given for a cluster, it indicates that no important features were found for that cluster.

### Feature List:
  - Lemmatized vocabulary
  - Dependency label of the target word
  - Morphological properties of the target word
  - Position of the target word in the sentence
  - Part-of-speech tag of the neighboring words of the target word

The features that are given between ** are for dependency labels (* _dobj_ * for direct object), position of the word in the sentence (* _pos_1_ * for the first word in the sentence), morphological properties of the word (* _NNS_ * for plural noun) and POS tag of the neighboring words (* _NUM_l_ * for number as left neighbor; _l_ for left neighbor, _r_ for right neighbor). The features are annotated automatically using the spaCy parser and tagger using the _en_core_web_sm model_ [link](https://spacy.io/models/en#en_core_web_sm).

The features that are given without * refer to the words/items that are found in the sentence. For example, _apple_ feature means that there is the word apple in the sentence.

