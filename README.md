# ContextDecode: Reverse Engineering for Automated Interpretation of Contextualized Embedding Clusters

ContextDecode is a powerful tool designed to automatically interpret clusters of contextualized word embeddings by identifying the sentence context features responsible for their formation.

## Purpose

The purpose of this tool is to provide automated interpretation of clusters of contextualized word embeddings. By analyzing the sentence context features that contribute to the formation of these clusters, ContextDecode offers valuable insights into the semantic nuances of words within different contexts.

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

