import argparse
from data_preparation.prepare_data import create_feature_matrix
from data_preparation.extract_embeddings import extract_embeddings_target_word
from feature_selection import select_best_features
from annotate_clusters import write_results, plot_clusters
from clustering import get_cluster_labels


"""
This script automatically annotates the clusters of a word based on the sentence 
features responsible for the formation of the clusters. It performs the following steps:

1. Extracts sentences containing the target word and its alternative forms.
2. Annotates these sentences with linguistic features (sentence context features).
3. Extracts embeddings for the target word in each sentence using a pre-trained language model.
4. Clusters the embeddings.
5. Selects the top features with recursive feature elimination.
6. Annotate the clusters with informative features and writes the results to files 
and plots the clusters with annotated features.

Usage:
    python main.py target_word alternative_forms n_sentences n_features

Arguments:
- target_word (str): Target word to analyze.
- alternative_forms (str): Alternative forms of the word, e.g., 
    "buy\bought" for "buy".
- n_sentences (int): Number of sentences to extract.
- n_features (int): Number of sentence features to select.

Example:
    python script.py buy buy\bought 200 50
"""


def main():
    parser = argparse.ArgumentParser(
        description="Automatically annotates the clusters of a word based on the"
        "sentence features resposible for the formation of the clusters."
    )
    parser.add_argument("target_word", type=str, help="Target word")
    parser.add_argument(
        "alternative_forms",
        type=str,
        help="Alternative forms of the word like 'buy' and 'bought' for 'buy'."
        "Give as: buy\\bought",
    )
    parser.add_argument("n_sentences", type=int, help="Number of sentences to extract")
    parser.add_argument(
        "n_features", type=int, help="Number of sentence features to select"
    )
    args = parser.parse_args()

    df = create_feature_matrix(
        args.target_word, args.alternative_forms, n_sentences=args.n_sentences
    )
    vectors = extract_embeddings_target_word(df)

    # Cluster and get cluster labels
    cluster_labels = get_cluster_labels(vectors)
    df.insert(2, "*cluster_label*", cluster_labels, True)

    # Feature Selection
    selected_features = select_best_features(df, n_features=args.n_features)

    # Write the result
    write_results(args.target_word, df, selected_features)
    plot_clusters(args.target_word, df, selected_features, vectors)

    print("Clusters are annotated. Results are ready in the 'results' folder.")


if __name__ == "__main__":
    main()
