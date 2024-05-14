import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import json
from sklearn.svm import LinearSVC
import pandas as pd
import random


"""Feature Selection with Recursive Feature Elimination (RFE)"""


def reduce_features(features_list, n=10):
    feature_indices = np.random.randint(low=0, high=len(features_list), size=n)
    reduced_feature_list = []
    for features in features_list:
        reduced_feature_list.append(np.array([features[i] for i in feature_indices]))
    return reduced_feature_list

def train_test_clf(split_dataset, clf):
    clf.fit(split_dataset["X_train"], split_dataset["y_train"])

    y_pred = []
    for i in range(len(split_dataset["X_test"])):
        feat = np.asarray(split_dataset["X_test"][i]).reshape(1, -1)
        prediction = clf.predict(feat)
        y_pred.append(prediction[0])

    f1_details = precision_recall_fscore_support(
        split_dataset["y_test"], y_pred, average="weighted", zero_division=0.0
    )
    # print(f'Precision: {f1_details[0]}, Recall: {f1_details[1]}, F-Beta: {f1_details[2]}')
    print(f'F-Beta: {f1_details[2]}')
    return clf


target_word = 'book'
feature_no = 25

df = pd.read_pickle(f'data/{target_word}_clustered.pickle')

labels = df['*cluster_labels*'].values
feature_matrices = df.drop(columns=df.columns[:4]).values
feature_names = df.drop(columns=df.columns[:4]).columns.tolist()
reduced_feature_matrices = reduce_features(feature_matrices, n=feature_no)


X_train, X_test, y_train, y_test = train_test_split(
    feature_matrices, labels, test_size=0.33, random_state=42
)
split_dataset = {
    "X_train": X_train,
    "y_train": y_train,
    "X_test": X_test,
    "y_test": y_test,
}

X_train, X_test, y_train, y_test = train_test_split(
    reduced_feature_matrices, labels, test_size=0.33, random_state=42
)
split_dataset_reduced = {
    "X_train": X_train,
    "y_train": y_train,
    "X_test": X_test,
    "y_test": y_test,
}

print('Baseline Classifier:')
clf_reduced = LinearSVC(random_state=0, tol=1e-5, max_iter=100000, C=100, dual=True, class_weight='balanced')
clf_reduced = train_test_clf(split_dataset_reduced, clf_reduced)

print(f'Top Features Classifier:')
clf = LinearSVC(random_state=0, tol=1e-5, max_iter=100000, C=100, dual=True, class_weight='balanced')
clf.fit(split_dataset["X_train"], split_dataset["y_train"])
selector = RFE(clf, n_features_to_select=feature_no, step=50)
selector = train_test_clf(split_dataset, selector)
feature_importance = selector.ranking_


# write selected features
best_features = []
for f_i in range(len(feature_importance)):
    if feature_importance[f_i] == 1:
        f_name = feature_names[f_i]
        best_features.append(f_name)

with open(f'results/best_{feature_no}_features_{target_word}.json', 'w') as outfile:
    json.dump(best_features, outfile)
        
