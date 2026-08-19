"""Machine-learning models implemented using only Python's standard library."""

import math


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
