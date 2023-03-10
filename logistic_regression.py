import numpy as np


def model(X, weights, bias):
    """
    Produces a prediction of y based on the given parameters using logistic regression

    Args:
      X (ndarray): Shape (n,m) array of m input vectors with n features
      weights (ndarray): Shape (n,) model parameters
      bias (scalar):  model parameter

    Returns:
      predictions (ndarray): Shape (m,) array of predictions for y
    """

    predictions = np.dot(weights, X) + bias

    for prediction in predictions:
        prediction = 1 / (1 + 1 / np.exp(prediction))

    return predictions


def logistic_cost(predictions, y):
    """
    Computes cost

    Args:
      predictions (ndarray (m,)): predicted values
      y (ndarray (m,)) : target values

    Returns:
      cost (scalar): cost
    """

    cost = 0.0

    for i in range(len(y)):
        cost += -y[i] * np.log(predictions[i]) - (1 - y[i]) * np.log(1 - predictions[i])

    cost /= len(y)
    return cost
