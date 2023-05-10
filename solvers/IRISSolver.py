import numpy as np
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import KFold

from normalization.normal import getStatisticalParameters, statisticalNormalisation, statisticalScalingParam
from regression.batch_logistic_regression import MyBatchLogisticRegression, sigmoid


def normalize(images, params = []):
    feature_matrix = [[] for i in range(len(images[0]))]
    for image in images:
        for i in range(len(image)):
            feature_matrix[i].append(image[i])
    param_list = []
    normalized_matrix = []
    if len(params) == 0 :
        for i in range(len(feature_matrix)):
            mean,dev = getStatisticalParameters(feature_matrix[i])
            param_list.append([mean,dev])
            normalized_matrix.append(statisticalNormalisation(feature_matrix[i]))
    else:
        for i in range(len(feature_matrix)):
            mean = params[i][0]
            dev = params[i][1]
            normalized_matrix.append(statisticalScalingParam(feature_matrix[i], mean, dev))
        param_list = params

    new_images = [[] for i in range(len(images))]
    for i in range(len(normalized_matrix)):
        for j in range(len(images)):
            new_images[j].append(normalized_matrix[i][j])

    return new_images, param_list


def recall(y_true, y_pred):
    tp = 0
    fn = 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            fn += 1

    if tp + fn > 0:
        recall_score = tp / (tp + fn)
    else:
        recall_score = 0.0

    return recall_score

def precision(y_true, y_pred):
    tp = sum([1 for i in range(len(y_true)) if y_true[i] == 1 and y_pred[i] == 1])
    fp = sum([1 for i in range(len(y_true)) if y_true[i] == 0 and y_pred[i] == 1])
    if tp + fp == 0:
        return 0
    return tp / (tp + fp)

def binary_cross_entropy(y_pred, y_true):
    #print(y_true, y_pred)
    loss = 0
    for i in range(len(y_true)):
        # Compute loss for each prediction/label pair
        loss += y_true[i] * np.log(y_pred[i]) + (1 - y_true[i]) * np.log(1 - y_pred[i])
    return -loss / len(y_true)

def iris(): # [0,1,2] <=> [setosa, versicolor, virginica]
    print("IRIS cod propriu :\n")

    data = load_iris()
    inputs = data['data']
    outputs = data['target']
    # print(data['target_names'])
    # print(data['feature_names'])

    kf = KFold(n_splits=4, shuffle=True)
    allInputs = inputs
    impartire = 1
    for train_index, test_index in kf.split(allInputs):
        print("\n",impartire, "st split")
        impartire+=1
        # impartire test / train # seed =
        # np.random.seed(np.random.choice([3,4,7,8]))
        # perm = np.random.permutation(len(inputs))
        # inputs = inputs[perm]
        # outputs = outputs[perm]
        # indexes = [i for i in range(len(inputs))]
        # trainSample = np.random.choice(indexes, int(0.8 * len(inputs)), replace=False)
        # validationSample = [i for i in indexes if not i in trainSample]
        # trainInputs = [inputs[i] for i in trainSample]
        # trainLabels = [outputs[i] for i in trainSample]
        # validationInputs = [inputs[i] for i in validationSample]
        # validationLabels = [outputs[i] for i in validationSample]
        trainInputs, validationInputs = allInputs[train_index], allInputs[test_index]
        trainLabels, validationLabels = outputs[train_index], outputs[test_index]

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
            #print(new_valid_outputs)
            print("Accuracy ( for feature ", i, " vs all ) : ", correct / len(new_valid_outputs) * 100, "%")
            print("Recall ( for feature ", i, " vs all ) : ", recall(new_valid_outputs, predicted))
            print("Precision ( for feature ", i, " vs all ) : ", precision(new_valid_outputs, predicted))
            print("Binary cross entropy ( for feature ", i, " vs all ) : ", binary_cross_entropy(regressor.sigs(validationInputs), new_valid_outputs))

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
