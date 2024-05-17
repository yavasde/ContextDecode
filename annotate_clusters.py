import collections
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import os


"""
Module for analyzing clusters and their features. It includes methods 
to group data points by clusters, calculate the number of important features, 
find example sentences with specific features, identify features that are 
unique to a single cluster, write interpretation results to files, and plot clusters 
annotated with important features.

Functions:

- group_data_points_by_clusters(X_tsne, cluster_labels): Groups data points 
    by their cluster labels.
- calculate_important_feature_number(cluster_feature_dict): Calculates the 
    total number of important features across clusters.
- find_examples(df, feature, cluster, n_examples=5): Retrieves example sentences 
    that contain a specific feature within a specified cluster.
- find_one_cluster_features(df, selected_features): Identifies features that 
    are predominantly found in a single cluster.
- write_results(target_word, df, selected_features): Writes the results of 
    the analysis, including features and examples, to a file.
- plot_clusters(target_word, df, selected_features, vectors): Generates a 
    t-SNE plot of the clusters annotated with important features.
"""


def group_data_points_by_clusters(X_tsne, cluster_labels):
    """
    Groups data points by clusters.

    Parameters:
    X_tsne (array-like): 2D array containing the transformed data points.
    cluster_labels (array-like): Array containing cluster labels for each data point.

    Returns:
    dict: Dictionary mapping cluster labels to data points.
    """
    data_groups = {}
    for label in set(cluster_labels):
        data_groups[label] = np.array(
            [
                X_tsne[i]
                for i in range(len(cluster_labels))
                if cluster_labels[i] == label
            ]
        )
    return data_groups


def calculate_important_feature_number(cluster_feature_dict):
    """
    Calculates the number of important features.

    Parameters:
    cluster_feature_dict (dict): Dictionary mapping clusters to features.

    Returns:
    int: Number of important features.
    """
    feature_count = 0
    for feature_list in cluster_feature_dict.values():
        feature_count += len(feature_list)
    return feature_count


def find_examples(df, feature, cluster, n_examples=5):
    """
    Finds example sentences with features from the cluster.

    Parameters:
    df (DataFrame): Input dataframe containing features and labels.
    feature (str): Feature to find examples for.
    cluster (int): Cluster label.
    n_examples (int): Number of examples to retrieve.

    Returns:
    list: List of example sentences.
    """
    feature_df = df[df[feature] == 1]
    feature_and_cluster_df = feature_df[feature_df["*cluster_label*"] == cluster]
    sentences = feature_and_cluster_df["*sentence_str*"].values
    return sentences[:n_examples]


def find_one_cluster_features(df, selected_features):
    """
    Finds features that are found only in one cluster.

    Parameters:
    df (DataFrame): Input dataframe containing features and labels.
    selected_features (list): List of selected features.

    Returns:
    dict: Dictionary mapping features to clusters if they are found only in one cluster.
    """
    cluster_feature_dict = collections.defaultdict(list)
    for feature in selected_features:
        cluster_label_list = [
            sentence["*cluster_label*"]
            for _, sentence in df[df[feature] == 1].iterrows()
        ]
        most_common_cluster = collections.Counter(cluster_label_list).most_common(1)
        if (len(cluster_label_list) * 75 // 100) <= most_common_cluster[0][1]:
            cluster_feature_dict[most_common_cluster[0][0]].append(feature)
    return cluster_feature_dict


def write_results(target_word, df, selected_features):
    """
    Writes interpretation results to files.

    Parameters:
    target_word (str): Target word.
    df (DataFrame): Input dataframe containing features and labels.
    selected_features (list): List of selected features.
    """
    one_cluster_features = find_one_cluster_features(df, selected_features)
    important_feature_count = calculate_important_feature_number(one_cluster_features)
    print(
        f"{important_feature_count} features found important out of {len(selected_features)}"
    )

    file_path = f"results/{target_word}"
    isExist = os.path.exists(file_path)
    if not isExist:
        os.makedirs(file_path)
    with open(f"{file_path}/{target_word}.txt", "w") as results_file:
        # Write features and clusters
        results_file.write("Features that are found in one cluster:\n")
        for cluster, feature_list in one_cluster_features.items():
            results_file.write(
                f"Cluster: {cluster+1}\tFeature: {', '.join(feature_list)}\n"
            )

        # Write examples
        results_file.write("\n-------------------\n\nExample:\n")
        for cluster, feature_list in one_cluster_features.items():
            for feature in feature_list:
                results_file.write(
                    f"\n\nFeature '{feature}' and Cluster {cluster+1}:\n"
                )
                examples = find_examples(df, feature, cluster, n_examples=5)
                for i in range(len(examples)):
                    results_file.write(f"{i+1}. {examples[i]}\n")


def plot_clusters(target_word, df, selected_features, vectors):
    """
    Plots clusters annotated with important features.

    Parameters:
    target_word (str): Target word.
    df (DataFrame): Input dataframe containing features and labels.
    selected_features (list): List of selected features.
    vectors (array-like): Array containing vectors.
    """
    one_cluster_features = find_one_cluster_features(df, selected_features)
    cluster_labels = df["*cluster_label*"].values

    # Perform TSNE
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(np.array(vectors))
    data_groups = group_data_points_by_clusters(X_tsne, cluster_labels)

    _, ax = plt.subplots()
    colors = [
        "c",
        "m",
        "y",
        "tab:orange",
        "tab:red",
        "tab:blue",
        "tab:purple",
        "tab:green",
        "k",
        "tab:brown",
    ]

    # Plot Data Points
    for label, data in data_groups.items():
        ax.scatter(
            data[:, 0],
            data[:, 1],
            c=colors[label],
            alpha=0.3,
            edgecolors="none",
            label=f"Cluster {label+1}",
        )

    # Annotate Features
    bbox = dict(boxstyle="square,pad=0.1", fc="w", ec="k")
    for cluster, feature_list in one_cluster_features.items():
        cluster_points = data_groups[cluster]
        coordinate = np.mean(cluster_points, axis=0)
        ax.annotate(
            "\n".join(feature_list),
            coordinate,
            textcoords="offset points",
            xytext=(5, 5),
            ha="left",
            fontsize=8,
            bbox=bbox,
        )

    plt.legend()
    plt.savefig(f"results/{target_word}/{target_word}_tsne.png")
