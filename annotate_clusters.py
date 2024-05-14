import json
import pandas as pd
import collections


target_word = 'book'
feat_no = 25
cluster_no = 5

with open(f"results/best_{feat_no}_features_{target_word}.json", "r") as feat_list_file:
    feature_list = json.load(feat_list_file)

df = pd.read_pickle(f'data/{target_word}_clustered.pickle')



cluster_features = {}
for feat in feature_list:
    cluster_features[feat] = []
    feat_df = df[df[feat] == 1]
    for data_id, d in feat_df.iterrows():
        cluster_features[feat].append(d["*cluster_labels*"])

d = {}
for cluster_label in range(cluster_no):
    d[cluster_label] = []
    for f_name, feat_cluster_labels in cluster_features.items():
        for cl in feat_cluster_labels:
            if cl == cluster_label:
                d[cluster_label].append(f_name)

one_cluster_only = collections.defaultdict(list)
# some feature is found in one cluster
for f_name, cluster_labels in cluster_features.items():
    most_common_label = collections.Counter(cluster_labels).most_common(1)
    if (len(cluster_labels)*80 // 100) <= most_common_label[0][1]:
        one_cluster_only[most_common_label[0][0]].append((f_name, most_common_label[0][1]))
        print(f_name, most_common_label[0][0])

#print common features for each cluster
#print features that are only found in one cluster

# one clusteris formed by


# for l, feat_list in d.items():
#     features = collections.Counter(feat_list)
#     print(l)
#     print(features)

#more than one feature


# sentence features
cluster_sentence_features = collections.defaultdict(list)
for data_id, d in df.iterrows():
    sent_features = []
    for feat in feature_list:
        if d[feat] == 1:
            sent_features.append(feat)
    cluster_sentence_features[d["*cluster_labels*"]].append({'sent_str': d["*sentence_str*"],
        'features': sent_features,
            })

# for cluster, sentences in cluster_sentence_features.items():
#     cluster_features =  [sent['features'] for sent in sentences]
#     frequent_features = collections.Counter([f2 for f in cluster_features for f2 in f])
#     print(frequent_features)
