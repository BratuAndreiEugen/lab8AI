import numpy as np
from sklearn.datasets import load_iris

from regression.batch_logistic_regression import MyBatchLogisticRegression, sigmoid
from solvers.IRISSolver import recall, precision, binary_cross_entropy


def old_iris(): # [0,1,2] <=> [setosa, versicolor, virginica]
    print("IRIS cod propriu :\n")

    data = load_iris()
    inputs = data['data']
    outputs = data['target']
    # print(data['target_names'])
    # print(data['feature_names'])

    # impartire test / train # seed =
    #np.random.seed(np.random.choice([3, 4, 7, 8]))
    perm = np.random.permutation(len(inputs))
    inputs = inputs[perm]
    outputs = outputs[perm]
    indexes = [i for i in range(len(inputs))]
    trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
    validationSample = [i for i in indexes if not i in trainSample]
    trainInputs = [inputs[i] for i in trainSample]
    trainLabels = [outputs[i] for i in trainSample]
    validationInputs = [inputs[i] for i in validationSample]
    validationLabels = [outputs[i] for i in validationSample]

    # print(trainInputs)
    # normalizare
    # trainInputs, param = normalize(trainInputs)
    # validationInputs, pr = normalize(validationInputs, param)

    # print(trainInputs[0])
    # print(trainLabels)
    # aplic LogReg pt fiecare label sub forma 1 vs. all
    target_coeficients = []
    for i in range(
            len(data['target_names'])):  # 0 - true ( are labelul curent ) 1 - false ( are oricare alt label )
        new_outputs = []
        for j in range(len(trainLabels)):
            if trainLabels[j] == i:
                new_outputs.append(1)
            else:
                new_outputs.append(0)
        regressor = MyBatchLogisticRegression()
        weights, intercept = regressor.fit(trainInputs, new_outputs)
        # print("W&I : ", weights, intercept)
        d = {'weights': weights, 'intercept': intercept}
        target_coeficients.append(d)

        new_valid_outputs = []
        percentage = 0
        for j in range(len(validationLabels)):
            if validationLabels[j] == i:
                new_valid_outputs.append(1)
                percentage += 1
            else:
                new_valid_outputs.append(0)

        percentage /= len(validationLabels)
        regressor.setThreshold(1 - percentage)

        predicted = regressor.predict(validationInputs)
        # print(regressor.sigs(validationInputs))
        correct = 0
        for j in range(len(predicted)):
            if predicted[j] == new_valid_outputs[j]:
                correct += 1
        # print(new_valid_outputs)
        print("Accuracy ( for feature ", i, " vs all ) : ", correct / len(new_valid_outputs) * 100, "%")
        print("Recall ( for feature ", i, " vs all ) : ", recall(new_valid_outputs, predicted))
        print("Precision ( for feature ", i, " vs all ) : ", precision(new_valid_outputs, predicted))
        print("Binary cross entropy ( for feature ", i, " vs all ) : ",
              binary_cross_entropy(regressor.sigs(validationInputs), new_valid_outputs))

    correct = 0
    p = 0
    # print(validationLabels)
    for inp in validationInputs:
        computed = []
        for i in range(len(data['target_names'])):
            # print("INP&W : ", inp)
            # print(target_coeficients[i]['weights'])
            f = sigmoid(target_coeficients[i]['intercept'] + np.dot(inp, target_coeficients[i]['weights']))
            computed.append(f)
        # print(computed)
        # print(validationLabels[p])
        if np.argmax(computed) == validationLabels[p]:
            correct += 1

        p += 1

    print("Accuracy : ", correct / len(validationInputs) * 100, "%")
