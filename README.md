# ContextDecode: Reverse Engineering Contextualized Embedding Clusters for Automated Interpretation

ContextDecode is a tool designed to automatically interpret clusters of contextualized word embeddings by identifying the sentence context features responsible for their formation.

## Method Details

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


Additionally, the performance of the classifier trained with the top-10 features is provided. This is compared to a baseline classifier trained with random 10 features, offering insights into the effectiveness of the selected features for interpreting the clusters.

If no features are given for a cluster, it indicates that no important features were found for that cluster.

### Feature List:
  - Lemmatized vocabulary
  - Dependency label of the target word
  - Morphological properties of the target word
  - Position of the target word in the sentence
  - Part-of-speech tag of the neighboring words of the target word

The features that are given between ** are for dependency labels (_*dobj*_ for direct object), position of the word in the sentence (_*pos_1*_ for the first word in the sentence), morphological properties of the word (_*NNS*_ for plural noun) and POS tag of the neighboring words (_*NUM_l*_ for number as left neighbor; l for left neighbor, r for right neighbor). The features are annotated automatically using the spaCy parser and tagger using the _en_core_web_sm model_ [link](https://spacy.io/models/en#en_core_web_sm).

The features that are given without * refer to the words/items that are found in the sentence. For example, _apple_ feature means that there is the word apple in the sentence.

