from statistics import mean


def minMaxScaling(features):
    minim = min(features)
    maxim = max(features)
    scaled_features = [(f - min(features))/(max(features) - min(features)) for f in features]
    return scaled_features

def statisticalNormalisation(features):
    meanValue = mean(features)
    stdDevValue = (1 / len(features) * sum([(feat - meanValue) ** 2 for feat in features])) ** 0.5
    if stdDevValue == 0:
        return features
    normalised_features = [(feat - meanValue) / stdDevValue for feat in features]
    return normalised_features

def getMinMaxParameters(features):
    return min(features), max(features)

def getStatisticalParameters(features):
    meanValue = mean(features)
    return meanValue, (1 / len(features) * sum([(feat - meanValue) ** 2 for feat in features])) ** 0.5

def minMaxScalingParam(features, minim, maxim):
    scaled_features = [(f - minim) / (maxim - minim) for f in features]
    return scaled_features

def statisticalScalingParam(features, meanValue, stdDev):
    if stdDev == 0:
        return features
    normalised_features = [(feat - meanValue) / stdDev for feat in features]
    return normalised_features