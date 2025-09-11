import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Question 1a
# generate data
X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=15, shuffle=False,
                           random_state=0)

# Normalize the data
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# shuffle the data
shuffle_idxs = np.random.default_rng(seed=0).permutation(X.shape[1])
X_shff = X[:, shuffle_idxs]

# Desicion Tree Classifier
clf = DecisionTreeClassifier(criterion='entropy', random_state=4)
clf.fit(X_shff, y)
importances = clf.feature_importances_
# descending order
indices = np.argsort(importances)[::-1][:5]  # top 5 features
# map the top 5 features to the original data
top5 = shuffle_idxs[indices]
# count the number of times each feature is in the top 5
# true_importance = np.arange(5)
# count = len(set(true_importance).intersection(set(indices)))
count = sum(1 for idx in top5 if idx < 5)
print(f"Number of true informative features in top 5: {count}")

# plot the top 5 features
plt.bar(range(len(importances)), importances[np.argsort(importances)[::-1]])
plt.xlabel('Feature Rank')
plt.ylabel('Feature Importance Score')
plt.title('Feature Importance Ranking')
plt.show()

# Q1(c)
counts = []
for i in range(1000):
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=15, shuffle=False,
                               random_state=i)

    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # shuffle the data with seed=i
    shuffle_idxs = np.random.default_rng(seed=i).permutation(X.shape[1])
    X_shff = X[:, shuffle_idxs]

    # Desicion Tree Classifier
    clf = DecisionTreeClassifier(criterion='entropy', random_state=4)
    clf.fit(X_shff, y)
    importances = clf.feature_importances_
    # descending order
    indices = np.argsort(importances)[::-1][:5]  # top 5 features
    # map the top 5 features to the original data

    top5 = shuffle_idxs[indices]
    # count the number of times each feature is in the top 5
    count = np.sum(top5 < 5)
    counts.append(count)

# plot
plt.hist(counts, bins=np.arange(0, 6) - 0.5, edgecolor='black')
plt.xticks(range(0, 6))
plt.xlabel('Number of True Features Recovered')
plt.ylabel('Frequency')
plt.title('Histogram of True Features Recovered (Decision Tree)')
plt.show()
average = np.mean(counts)
print(f"Average number of good feature recovered:{average}")

# Q1(d)
# without scaling
counts_nscale = []
# repeat Q1c
for i in range(1000):
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=15, shuffle=False,
                               random_state=i)

    # shuffle the data with seed=i
    shuffle_idxs = np.random.default_rng(seed=i).permutation(X.shape[1])
    X_shff = X[:, shuffle_idxs]

    # Logistic Regression
    lr = LogisticRegression(penalty=None, max_iter=1000, random_state=4)
    lr.fit(X_shff, y)
    coef = np.abs(lr.coef_[0])
    # descending order
    indices = np.argsort(coef)[::-1][:5]  # top 5 features
    # map the top 5 features to the original data
    top5 = shuffle_idxs[indices]
    # count the number of times each feature is in the top 5
    count = np.sum(top5 < 5)
    counts_nscale.append(count)

# with scale
counts_scale = []
for i in range(1000):
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=15, shuffle=False,
                               random_state=i)

    scaler = StandardScaler()  # Normalisation
    X = scaler.fit_transform(X)
    # shuffle the data with seed=i
    shuffle_idxs = np.random.default_rng(seed=i).permutation(X.shape[1])
    X_shff = X[:, shuffle_idxs]
    # Logistic Regression
    lr = LogisticRegression(penalty=None, max_iter=1000, random_state=4)
    lr.fit(X_shff, y)
    coef = np.abs(lr.coef_[0])
    # descending order
    indices = np.argsort(coef)[::-1][:5]  # top 5 features
    # map the top 5 features to the original data
    top5 = shuffle_idxs[indices]
    # count the number of times each feature is in the top 5
    count = np.sum(top5 < 5)
    counts_scale.append(count)

# plot with scale
plt.hist(counts_nscale, bins=np.arange(0, 6) - 0.5, alpha=0.5, label='Unscaled')
plt.hist(counts_scale, bins=np.arange(0, 6) - 0.5, alpha=0.5, label='Scaled')
plt.legend()
plt.xlabel('Number of True Features Recovered')
plt.ylabel('Frequency')
plt.title('Histogram of True Features Recovered (Logistic Regression)')
plt.show()
average_nscale = np.mean(counts_nscale)
average_scale = np.mean(counts_scale)
print(f"Average number of good feature recovered(LR without scale):{average_nscale}")
print(f"Average number of good feature recovered(LR with scale):{average_scale}")

# Q1(f)
overlaps = []
for i in range(1000):
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=15, shuffle=False,
                               random_state=i)
    # Normalize the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # shuffle the data with seed=i
    shuffle_idxs = np.random.default_rng(seed=i).permutation(X.shape[1])
    X_shff = X[:, shuffle_idxs]
    # Desicion Tree Classifier
    dtc = DecisionTreeClassifier(criterion='entropy', random_state=4)
    dtc.fit(X_shff, y)
    importances = clf.feature_importances_
    # descending order
    indices = np.argsort(importances)[::-1][:5]  # top 5 features
    # map the top 5 features to the original data
    top5_tree = shuffle_idxs[indices]
    # Logistics Regression
    lr = LogisticRegression(penalty=None, random_state=4)
    lr.fit(X_shff, y)
    coef = np.abs(lr.coef_[0])
    # descending order
    indices = np.argsort(coef)[::-1][:5]  # top 5 features
    top5_lr = shuffle_idxs[indices]
    overlaps.append(len(set(top5_tree) & set(top5_lr)))
# plot
plt.hist(overlaps, bins=np.arange(7) - 0.5, edgecolor='black')
plt.xlabel('Number of Overlapping Features')
plt.title('Overlap Between Decision Tree and Logistic Regression')
plt.show()

# Q2(b)
# backward elimination algorithm
from sklearn.metrics import accuracy_score


def backward_elimination(n_feature_keep=5, seed=0):
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=15, shuffle=False,
                               random_state=seed)
    # Normalize the data
    X = scaler.fit_transform(X)
    # shuffle the data with seed
    shuffle_idxs = np.random.default_rng(seed=seed).permutation(X.shape[1])
    X_shff = X[:, shuffle_idxs]
    features = list(range(X_shff.shape[1]))
    while len(features) > n_feature_keep:
        scores = []
        for f in features:
            X_subset = X[:, [x for x in features if x != f]]
            lr = LogisticRegression(penalty=None, random_state=4)
            lr.fit(X_subset, y)
            scores.append(accuracy_score(y, lr.predict(X_subset)))
        smallest_feature = features[
            np.argmin(scores)]  # find the feature corresponding to th smallest drop in the metric
        features.remove(smallest_feature)
    origin_left = shuffle_idxs[features]
    correct = np.sum(origin_left < 5)
    print(f"Left features:{origin_left}")
    return correct


correct = backward_elimination(n_feature_keep=5, seed=0)
print(f" Number of correct features: {correct}")

# Q2(c)
recovered_f = []
for i in range(1000):
    correct = backward_elimination(n_feature_keep=5, seed=i)
    recovered_f.append(correct)
plt.hist(recovered_f, bins=np.arange(0, 6) - 0.5, edgecolor='black')
plt.xlabel('Number of True Features Recovered(Backward Elimination)')
plt.ylabel('Frequency')
plt.title('Backward Elimination: True Features Recovered (Logistic Regression)')
plt.show()
average_re = np.mean(recovered_f)
print(f"Average number of recovered feature (Backward Elimination):{average_re}")

# Q2(e)
from itertools import combinations


def best_subset(seed=0):
    X, y = make_classification(n_samples=1000, n_features=7, n_informative=3, n_redundant=4, shuffle=False,
                               random_state=seed)
    # Normalize the data
    X = scaler.fit_transform(X)
    # shuffle the data with seed
    shuffle_idxs = np.random.default_rng(seed=seed).permutation(X.shape[1])
    X_shff = X[:, shuffle_idxs]
    best_score = -np.inf
    best_subset = None

    for subset in combinations(range(X_shff.shape[1]), 3):
        X_subset = X_shff[:, subset]
        lr = LogisticRegression(penalty=None, random_state=4)
        lr.fit(X_subset, y)
        acc = accuracy_score(y, lr.predict(X_subset))
        if acc > best_score:
            best_score = acc
            best_subset = subset
    origin_subset = [shuffle_idxs[idx] for idx in best_subset]
    correct_best = sum(1 for idx in origin_subset if idx < 3)
    return correct_best


# repeat 1000
recoveries = []
for i in range(1000):
    correct_best = best_subset(seed=i)
    recoveries.append(correct_best)
plt.hist(recoveries, bins=np.arange(0, 6) - 0.5, edgecolor='black')
plt.xlabel('Number of Correct Features')
plt.title('Best Subset Selection')
plt.show()
print(f"Average correct(Best Selection):{np.mean(recoveries)}")

# Q2(f)
# Permutation Feature Importance score
from sklearn.inspection import permutation_importance


def permutation_importance_trial(seed=0):
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=15, shuffle=False,
                               random_state=seed)
    # Normalize the data
    X = scaler.fit_transform(X)
    # shuffle the data with seed
    shuffle_idxs = np.random.default_rng(seed=seed).permutation(X.shape[1])
    X_shff = X[:, shuffle_idxs]
    lr = LogisticRegression(penalty=None, random_state=4)
    lr.fit(X_shff, y)
    # calculate permutaion importance
    importance = permutation_importance(lr, X_shff, y, n_repeats=10, random_state=4)
    mean_importance = importance.importances_mean
    # top5
    indices = np.argsort(mean_importance)[::-1][:5]
    top5 = shuffle_idxs[indices]
    correct_pe = np.sum(top5 < 5)
    return correct_pe


count_pe = []
for i in range(1000):
    correct_pe = permutation_importance_trial(seed=i)
    count_pe.append(correct_pe)
# plot
plt.hist(count_pe, bins=np.arange(0, 6) - 0.5, edgecolor='black')
plt.xlabel('Number of True Feature')
plt.title('Permutation Importance')
plt.show()
print(f"Average correct:{np.mean(count_pe)}")