import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


def iris_tool():
    print("IRIS tools :\n")

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

    # Normalize data
    scaler = StandardScaler()
    train_inputs = scaler.fit_transform(trainInputs)
    validation_inputs = scaler.transform(validationInputs)

    # Train logistic regression model
    model = LogisticRegression(multi_class='ovr')
    model.fit(train_inputs, trainLabels)

    # Make predictions on validation set
    predicted_labels = model.predict(validation_inputs)
    print(predicted_labels)
    print(validationLabels)

    # Calculate accuracy
    accuracy = accuracy_score(validationLabels, predicted_labels)
    print("Accuracy : ", accuracy * 100, "%")