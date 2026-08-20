"""Machine-learning models implemented using only Python's standard library."""

import math
import random


class LinearRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        rows, cols = len(X), len(X[0])
        self.w = [0.0] * cols

        for _ in range(self.epochs):
            pred = self.predict(X)
            dw = [
                sum(X[i][j] * (pred[i] - y[i]) for i in range(rows))
                / rows
                for j in range(cols)
            ]
            db = sum((pred[i] - y[i]) for i in range(rows)) / rows
            self.w = [self.w[j] - self.lr * dw[j] for j in range(cols)]
            self.b -= self.lr * db

    def predict(self, X):
        rows, cols = len(X), len(X[0])
        return [
            sum(X[i][j] * self.w[j] for j in range(cols)) + self.b
            for i in range(rows)
        ]

    def errors(self, X, y):
        pred = self.predict(X)
        return [y[i] - pred[i] for i in range(len(X))]

    def evaluate(self, X, y):
        errors = self.errors(X, y)
        return sum(error * error for error in errors) / len(X)


class LogisticRegression:
    def __init__(self, lr=0.01, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = 0.0

    def sigmoid(self, z):
        if z >= 0:
            return 1 / (1 + math.exp(-z))

        exp_z = math.exp(z)
        return exp_z / (1 + exp_z)

    def fit(self, X, y):
        rows, cols = len(X), len(X[0])
        self.w = [0.0] * cols

        for _ in range(self.epochs):
            pred = self.predict_proba(X)
            dw = [
                sum(X[i][j] * (pred[i] - y[i]) for i in range(rows))
                / rows
                for j in range(cols)
            ]
            db = sum((pred[i] - y[i]) for i in range(rows)) / rows
            self.w = [self.w[j] - self.lr * dw[j] for j in range(cols)]
            self.b -= self.lr * db

    def predict_proba(self, X):
        rows, cols = len(X), len(X[0])
        return [
            self.sigmoid(sum(X[i][j] * self.w[j] for j in range(cols)) + self.b)
            for i in range(rows)
        ]

    def predict(self, X, threshold=0.5):
        pred = self.predict_proba(X)
        return [1 if pred[i] >= threshold else 0 for i in range(len(X))]

    def evaluate(self, X, y):
        pred = self.predict_proba(X)
        eps = 1e-15
        return -sum(
            y[i] * math.log(max(eps, min(1 - eps, pred[i])))
            + (1 - y[i])
            * math.log(1 - max(eps, min(1 - eps, pred[i])))
            for i in range(len(X))
        ) / len(X)


class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    @staticmethod
    def distance(first, second):
        """Return the squared Euclidean distance between two samples."""
        return sum(
            (first_value - second_value) ** 2
            for first_value, second_value in zip(first, second)
        )

    def get_neighbors(self, x):
        """Return the indices of the nearest training samples."""
        distances = [
            (index, self.distance(sample, x))
            for index, sample in enumerate(self.X_train)
        ]
        distances.sort(key=lambda item: item[1])
        return [index for index, _ in distances[: self.k]]

    def predict_one(self, x):
        """Predict one sample using majority vote, breaking ties by label."""
        neighbors = self.get_neighbors(x)
        labels = [self.y_train[i] for i in neighbors]
        counts = {}
        for label in labels:
            counts[label] = counts.get(label, 0) + 1
        highest_count = max(counts.values())
        return min(
            label for label, count in counts.items() if count == highest_count
        )

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    def evaluate(self, X, y):
        """Return classification accuracy."""
        pred = self.predict(X)
        return sum(
            predicted == actual
            for predicted, actual in zip(pred, y)
        ) / len(y)


class DecisionTreeClassifier:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.root = None

    @staticmethod
    def gini(y):
        if not y:
            return 0.0

        p = sum(1 for label in y if label == 1) / len(y)
        return 2 * p * (1 - p)

    @staticmethod
    def majority_label(y):
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        return min(counts, key=lambda label: (-counts[label], label))

    def best_split(self, X, y):
        rows, cols = len(X), len(X[0])
        best_gini = float("inf")
        best = None
        for j in range(cols):
            values = sorted(set(X[i][j] for i in range(rows)))
            thresholds = [
                (values[i - 1] + values[i]) / 2
                for i in range(1, len(values))
            ]
            for threshold in thresholds:
                y_left = [
                    y[i] for i in range(rows) if X[i][j] <= threshold
                ]
                y_right = [
                    y[i] for i in range(rows) if X[i][j] > threshold
                ]
                if not y_left or not y_right:
                    continue
                weighted_gini = (
                    len(y_left) / rows * self.gini(y_left)
                    + len(y_right) / rows * self.gini(y_right)
                )
                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best = (j, threshold)

        return best

    def build_tree(self, X, y, depth=0):
        # Stop if node is pure
        if len(set(y)) == 1:
            return {"label": y[0]}

        # Stop if max depth reached
        if depth >= self.max_depth:
            return {"label": self.majority_label(y)}

        split = self.best_split(X, y)

        # Stop if no valid split exists
        if split is None:
            return {"label": self.majority_label(y)}

        feature, threshold = split

        X_left = []
        y_left = []
        X_right = []
        y_right = []

        for i in range(len(X)):
            if X[i][feature] <= threshold:
                X_left.append(X[i])
                y_left.append(y[i])
            else:
                X_right.append(X[i])
                y_right.append(y[i])

        if not X_left or not X_right:
            return {"label": self.majority_label(y)}

        return {
            "feature": feature,
            "threshold": threshold,
            "left": self.build_tree(X_left, y_left, depth + 1),
            "right": self.build_tree(X_right, y_right, depth + 1),
        }

    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict_one(self, x):
        node = self.root
        while "label" not in node:
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]

        return node["label"]

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    def evaluate(self, X, y):
        pred = self.predict(X)
        return sum(
            predicted == actual
            for predicted, actual in zip(pred, y)
        ) / len(y)


class BaggingClassifier:
    def __init__(self, n_estimators=10, max_depth=3):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.models = []

    @staticmethod
    def bootstrap(X, y):
        n = len(X)
        indices = [random.randrange(n) for _ in range(n)]
        X_sample = [X[i] for i in indices]
        y_sample = [y[i] for i in indices]
        return X_sample, y_sample

    def fit(self, X, y):
        self.models = []

        for _ in range(self.n_estimators):
            X_sample, y_sample = self.bootstrap(X, y)

            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.models.append(tree)

    def predict_one(self, x):
        predictions = [tree.predict_one(x) for tree in self.models]
        counts = {}
        for label in predictions:
            counts[label] = counts.get(label, 0) + 1
        return max(counts, key=counts.get)

    def predict(self, X):
        return [self.predict_one(x) for x in X]


class KMeans:
    def __init__(self, k=2, max_iters=100):
        self.k = k
        self.max_iters = max_iters
        self.centroids = None

    @staticmethod
    def distance(first, second):
        """Return the squared Euclidean distance between two samples."""
        return sum(
            (first_value - second_value) ** 2
            for first_value, second_value in zip(first, second)
        )

    def assign_clusters(self, X):
        # Return one cluster index per row in X
        clusters = []

        for x in X:
            min_distance = float("inf")
            cluster_idx = None

            for idx, centroid in enumerate(self.centroids):
                distance = self.distance(x, centroid)

                if distance < min_distance:
                    min_distance = distance
                    cluster_idx = idx

            clusters.append(cluster_idx)

        return clusters

    def recompute_centroids(self, X, labels):
        # Return updated centroid list
        # If a cluster is empty, keep its previous centroid
        rows, cols = len(X), len(X[0])
        centroids = []

        for k in range(self.k):
            X_k = [X[i] for i in range(rows) if labels[i] == k]

            if not X_k:
                centroids.append(self.centroids[k])
            else:
                centroids.append(
                    [
                        sum(X_k[i][j] for i in range(len(X_k))) / len(X_k)
                        for j in range(cols)
                    ]
                )

        return centroids

    def fit(self, X):
        # Initialize centroids using the first k points
        # Alternate assignment and centroid update
        # Stop early if labels stop changing
        self.centroids = [row[:] for row in X[: self.k]]
        prev_clusters = None

        for _ in range(self.max_iters):
            clusters = self.assign_clusters(X)

            if clusters == prev_clusters:
                break

            self.centroids = self.recompute_centroids(X, clusters)
            prev_clusters = clusters

    def predict(self, X):
        # Return nearest-centroid cluster for each row
        return self.assign_clusters(X)
