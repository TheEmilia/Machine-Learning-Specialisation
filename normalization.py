import numpy as np


def feature_scale(features):
    # x_i/max
    scaled_features = []
    for feature in features:
        scaled_features.append((feature / max(features)))

    return scaled_features


def mean_normalization(features):
    # (x_i - mean_i) / (max-min)
    scaled_features = []
    feature_mean = np.mean(features)
    feature_range = max(features) - min(features)

    for feature in features:
        scaled_features.append(((feature - feature_mean) / feature_range))

    return scaled_features


def zscore_normalization(features):
    # (x_i - mean_i) / std. deviation
    scaled_features = []
    feature_mean = np.mean(features)
    std_dev = np.std(features)

    for feature in features:
        scaled_features.append(((feature - feature_mean) / (std_dev)))

    return scaled_features
