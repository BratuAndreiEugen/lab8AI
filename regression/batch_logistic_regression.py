from math import exp
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class MyBatchLogisticRegression:
    def __init__(self):
        self.__w = None
        self.__inter = 0
        self.__threshold = 0.4

    def fit2(self, x, y, learning_rate=0.001, iterations=1000):
        self.coef_ = [0.0 for _ in range(1 + len(x[0]))]  # beta or w coefficients y = w0 + w1 * x1 + w2 * x2 + ...
        # self.coef_ = [random.random() for _ in range(len(x[0]) + 1)]    #beta or w coefficients
        for epoch in range(iterations):
            # TBA: shuffle the trainind examples in order to prevent cycles
            for i in range(len(x)):  # for each sample from the training data
                ycomputed = sigmoid(self.eval(x[i], self.coef_))  # estimate the output
                crtError = ycomputed - y[i]  # compute the error for the current sample
                for j in range(0, len(x[0])):  # update the coefficients
                    self.coef_[j + 1] = self.coef_[j + 1] - learning_rate * crtError * x[i][j]
                self.coef_[0] = self.coef_[0] - learning_rate * crtError * 1

        self.__intercept = self.coef_[0]
        self.__w= self.coef_[1:]

        return self.__w, self.__intercept


    def fit(self, features, output, learning_rate=0.01, iterations=1000, batch_size=32):
        num_features = len(features[0])
        weights = np.zeros(num_features)
        intercept = 0.0
        # print("feat : ", features)
        # print("op : ", output)
        # set number of batches
        batches = len(features) // batch_size

        for i in range(iterations):
            # for each batch,
            for j in range(batches):
                batch_features = features[j * batch_size: (j + 1) * batch_size]
                batch_output = output[j * batch_size: (j + 1) * batch_size]
                pred = sigmoid(np.dot(batch_features, weights) + intercept)

                # compute the gradients of the loss for weights and intercept
                batch_features = np.array(batch_features)
                error = pred - batch_output

                dw = (1 / batch_size) * np.dot(batch_features.T, error)
                di = (1 / batch_size) * np.sum(error)

                # intercept and weights update
                weights = weights - learning_rate * dw
                intercept = intercept - learning_rate * di

        # print(weights)
        self.__w = weights
        self.__inter = intercept
        return weights, intercept

    def fit_hinge(self, features, output, learning_rate=0.01, iterations=1000, batch_size=32):
        num_features = len(features[0])
        weights = np.zeros(num_features)
        intercept = 0.0
        # set number of batches
        batches = len(features) // batch_size

        for i in range(iterations):
            # for each batch,
            for j in range(batches):
                batch_features = features[j * batch_size: (j + 1) * batch_size]
                batch_output = output[j * batch_size: (j + 1) * batch_size]
                pred = np.dot(batch_features, weights) + intercept

                # compute the gradients of the loss for weights and intercept
                batch_features = np.array(batch_features)
                error = np.maximum(0, 1 - batch_output * pred) > 0
                indices = np.where(error)[0]

                dw = np.dot(batch_features[indices].T, -batch_output[indices]) / batch_size
                di = -np.sum(batch_output[indices]) / batch_size

                # intercept and weights update
                weights -= learning_rate * dw
                intercept -= learning_rate * di

        self.__w = weights
        self.__inter = intercept
        return weights, intercept

    def predictOneSample(self, sampleFeatures):
        coefficients = [self.__inter] + [c for c in self.__w]
        computedFloatValue = np.dot(sampleFeatures, coefficients[1:]) + coefficients[0]
        computed01Value = sigmoid(computedFloatValue)
        computedLabel = 0 if computed01Value < self.__threshold else 1
        return computedLabel

    def predict(self, inTest):
        computedLabels = [self.predictOneSample(sample) for sample in inTest]
        return computedLabels

    def eval(self, xi, coef):
        yi = coef[0]
        for j in range(len(xi)):
            yi += coef[j + 1] * xi[j]
        return yi

    def sigs(self, inTest):
        coefficients = [self.__inter] + [c for c in self.__w]
        return [sigmoid(self.eval(sample, coefficients)) for sample in inTest]

    def setThreshold(self, value):
        self.__threshold = value