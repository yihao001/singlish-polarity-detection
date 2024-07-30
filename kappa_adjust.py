import csv
import pickle

import numpy as np

from datalabel import label_data

TARGET_DIRECTORY = './'

POSITIVE_THRESHOLD_VALUES = np.linspace(0,1,11)
NEGATIVE_THRESHOLD_VALUES = np.linspace(0,1,11)

mapping = {
    'positive': 0,
    'neutral': 1,
    'negative': 2
}

def build_weight_matrix(categories, mode):
    if mode == 'unweighted':
        # [[0, 1, 1],
        #  [1, 0, 1],
        #  [1, 1, 0]]
        return np.fromiter((i != j
            for i in range(categories)
            for j in range(categories)), np.int).reshape(categories, -1)
    elif mode == 'squared':
        # [[0, 1, 4],
        #  [1, 0, 1],
        #  [4, 1, 0]]
        return np.fromiter((abs(i - j) ** 2
            for i in range(categories)
            for j in range(categories)), np.int).reshape(categories, -1)
    else: # linear
        # [[0, 1, 2],
        #  [1, 0, 1],
        #  [2, 1, 0]]
        return np.fromiter((abs(i - j)
            for i in range(categories)
            for j in range(categories)), np.int).reshape(categories, -1)

def build_observed_matrix(categories, subjects, ratings):
    observed = np.zeros((categories, categories))
    for k in range(subjects):
        observed[ratings[k, 0], ratings[k, 1]] += 1

    return observed / subjects

def build_distributions_matrix(categories, subjects, ratings):
    distributions = np.zeros((categories, 2))
    for k in range(subjects):
        distributions[ratings[k, 0], 0] += 1
        distributions[ratings[k, 1], 1] += 1

    return distributions / subjects

def build_expected_matrix(categories, distributions):
    return np.fromiter((distributions[i, 0] * distributions[j, 1]
        for i in range(categories)
        for j in range(categories)), np.float).reshape(categories, -1)

def calculate_kappa(weighted, observed, expected):
    sum_expected = sum(sum(weighted * expected))
    return 1.0 - ((sum(sum(weighted * observed)) / sum_expected) if sum_expected != 0 else 0.0)

kappa_scores = []

manual_labels = []
with open('./data/kappa/labels_kappa.csv') as f:
    org = csv.reader(f, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL)
    for l in org:
        manual_labels.append(l[2])

for pos in POSITIVE_THRESHOLD_VALUES:
    for neg in NEGATIVE_THRESHOLD_VALUES:

        labels = label_data(pos, neg)
        manual_numerical_labels_300 = [mapping[label] for label in manual_labels[:300]] # list of 0,1,2
        auto_numerical_labels_300 = [mapping[label] for label in labels[:300]] # list of 0,1,2
        
        ratings = []
        for man, auto in zip(manual_numerical_labels_300, auto_numerical_labels_300):
            ratings.append([auto,man])

        ratings = np.array(ratings)

        mode = 'unweighted'

        categories = int(np.amax(ratings)) + 1
        subjects = int(ratings.size / 2)
        weighted = build_weight_matrix(categories, mode)
        observed = build_observed_matrix(categories, subjects, ratings)
        distributions = build_distributions_matrix(categories, subjects, ratings)
        expected = build_expected_matrix(categories, distributions)
        kappa = calculate_kappa(weighted, observed, expected)

        print('Kappa (' + mode + ') for pos/neg ' + str(pos) + ', ' + str(neg) + ': ' + str(kappa))

        kappa_scores.append((pos, neg, kappa))

kappa_scores = sorted(kappa_scores, key=lambda x: x[2])

for score in kappa_scores:
    print(score)

f = open(TARGET_DIRECTORY + "kappa_scores.pkl","wb")
pickle.dump(kappa_scores,f)
f.close()

