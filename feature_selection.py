import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC


"""Module for Feature Selection with Recursive Feature Elimination (RFE)

This module provides functions for feature selection using Recursive Feature 
Elimination (RFE). It includes functions to reduce the number of features in 
a list of feature matrices randomly for the baseline classifier, prepare datasets 
for training and testing, test classifiers, train a random baseline classifier,
train a classifier and obtain the feature selector, and select the best features.

Functions:
- reduce_features: Reduces the number of features in a list of feature matrices randomly.
- dataset_preparation: Prepares the dataset for training and testing.
- test_clf: Tests the classifier on the test dataset and returns the weighted F1 score.
- train_baseline_classifier: Trains a random baseline classifier for 5 times and prints 
    the average F1 score.
- train_and_get_selector: Trains a classifier and returns the feature selector.
- select_best_features: Selects the best features from the selector and returns them as a list.
"""


def reduce_features(features_list, n_features=10):
    """
    Reduces the number of features in a list of feature matrices randomly.

    Parameters:
    features_list (list of arrays): List containing feature matrices.
    n_features (int): Number of features to select.

    Returns:
    list of arrays: Reduced feature matrices.
    """
    random_feature_indices = np.random.randint(
        low=0, high=len(features_list), size=n_features
    )
    reduced_feature_list = [
        np.array([features[i] for i in random_feature_indices])
        for features in features_list
    ]
    return reduced_feature_list


def dataset_preparation(df, random=False, n_features=10):
    """
    Prepares the dataset for training and testing.

    Parameters:
    df (DataFrame): Input dataframe containing features and labels.
    random (bool): Whether to randomly select features.
    n_features (int): Number of features to select.

    Returns:
    dict: Dictionary containing train and test datasets.
    """
    cluster_labels = df["*cluster_label*"].values
    feature_matrices = df.drop(columns=df.columns[:4]).values
    if random:
        feature_matrices = reduce_features(feature_matrices, n_features=n_features)

    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrices, cluster_labels, test_size=0.33, random_state=42
    )
    dataset = {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }
    return dataset


def test_clf(X_test, y_test, clf):
    """
    Test the classifier on the test dataset and return the weighted F1 score.

    Parameters:
    X_test (array-like): Test features.
    y_test (array-like): True labels for the test set.
    clf (estimator object): Trained classifier.

    Returns:
    float: Weighted F1 score.
    """
    y_pred = []
    for i in range(len(X_test)):
        sentence_features = np.asarray(X_test[i]).reshape(1, -1)
        prediction = clf.predict(sentence_features)
        y_pred.append(prediction[0])

    f1_details = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0.0
    )
    return f1_details[2]


def train_baseline_classifier(df, n_features=10):
    """
    Trains a random baseline classifier for 5 times and prints the average F1 score.

    Parameters:
    df (DataFrame): Input dataframe containing features and labels.
    n_features (int): Number of features to select.
    """
    n_iteration = 5
    fscores = 0
    for i in range(n_iteration):
        dataset_random = dataset_preparation(df, random=True, n_features=n_features)
        clf = LinearSVC(
            random_state=0,
            tol=1e-5,
            max_iter=100000,
            C=100,
            dual=True,
            class_weight="balanced",
        )
        clf.fit(dataset_random["X_train"], dataset_random["y_train"])
        fscores += test_clf(dataset_random["X_test"], dataset_random["y_test"], clf)
    average_fscore = np.sum(fscores) / n_iteration
    print(average_fscore)


def train_and_get_selector(df, n_features=10):
    """
    Trains a classifier and returns the feature selector.

    Parameters:
    df (DataFrame): Input dataframe containing features and labels.
    n_features (int): Number of features to select.

    Returns:
    object: Feature selector.
    """
    dataset = dataset_preparation(df, n_features=n_features)
    clf = LinearSVC(
        random_state=0,
        tol=1e-5,
        max_iter=100000,
        C=100,
        dual=True,
        class_weight="balanced",
    )
    clf.fit(dataset["X_train"], dataset["y_train"])
    selector = RFE(clf, n_features_to_select=n_features, step=10)
    selector.fit(dataset["X_train"], dataset["y_train"])
    print(test_clf(dataset["X_test"], dataset["y_test"], selector))
    return selector


def select_best_features(df, n_features=10):
    """
    Selects the best features from the selector and returns them as a list.

    Parameters:
    df (DataFrame): Input dataframe containing features and labels.
    n_features (int): Number of features to select.

    Returns:
    list: List of selected features.
    """
    print("TRAINING CLASSIFIERS\nBaseline Classifier Performance:")
    train_baseline_classifier(df, n_features=n_features)

    print(f"Top {n_features} Features Classifier Performance:")
    selector = train_and_get_selector(df, n_features=n_features)
    feature_importance_list = selector.ranking_
    feature_names = df.drop(columns=df.columns[:4]).columns.tolist()
    best_features = [
        feature_names[feature_index]
        for feature_index in range(len(feature_importance_list))
        if feature_importance_list[feature_index] == 1
    ]
    print(f"{len(best_features)} features selected out of {len(feature_names)}")
    return best_features
