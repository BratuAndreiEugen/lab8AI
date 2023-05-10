import numpy as np
from sklearn.datasets import load_iris

from regression.batch_logistic_regression import MyBatchLogisticRegression


def t():
    data = load_iris()
    inputs = data['data']
    outputs = data['target']

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

    sigs = []
    for i in range(len(data['target_names'])):
        classifier = MyBatchLogisticRegression()
        binaryTrainOutputs = [1 if trainOutput == i else 0 for trainOutput in trainLabels]
        classifier.fit(trainInputs, binaryTrainOutputs)

        sigs.append(classifier.sigs(validationInputs))

    computedTestOutputs = []
    # for every input
    for j in range(len(validationInputs)):
        bestClass = 0
        # for every output associated with the binary problem for the given class
        for i in range(len(sigs)):
            if sigs[i][j] > sigs[bestClass][j]:
                bestClass = i
        computedTestOutputs.append(bestClass)

    from sklearn.metrics import accuracy_score
    print("classification error (mine): ", accuracy_score(validationLabels, computedTestOutputs))
