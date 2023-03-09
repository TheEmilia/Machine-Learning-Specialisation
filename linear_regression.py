import numpy as np

# Multivariate Model


def model(X, weights, bias):
    """
    Produces a prediction of y based on the given parameters using linear regression

    Args:
      X (ndarray): Shape (n,m) array of m input vectors with n features
      weights (ndarray): Shape (n,) model parameters
      bias (scalar):  model parameter

    Returns:
      predictions (ndarray): Shape (m,) array of predictions for y
    """

    predictions = np.dot(weights, X) + bias

    return predictions


def mean_squared_error(predictions, y):
    """
    Calculates the MSE of the given parameters using the cost function

    Args:
      predictions (ndarray): Shape (m,) array of predictions for y
      y (ndarray): Shape (m,) true values of y

    Returns:
      mse (scalar): mean squared error of the model with the given parameters
    """

    mse = 0
    for i in range(len(y)):
        mse += (predictions[i] - y[i]) ** 2

    mse /= len(y)
    return mse


def update_model_parameters(X, weights, bias, predictions, y, learning_rate):
    """
    Calculates new values of the model parameters using gradient descent by way of the derivative of the cost function

    Args:
      X (ndarray): Shape (n,m) array of m input vectors with n features
      weights (ndarray): Shape (n,) model parameters
      bias (scalar):  model parameter
      predictions (ndarray): Shape (m,) array of predictions for y
      y (ndarray): Shape (m,) true values of y
      learning_rate (scalar): the magnitude by which to modify the model parameters

    Returns:
      weights (ndarray): Shape (n,) updated model parameters
      bias (scalar): updated model parameter
    """

    dj_dw = np.zeros(weights.shape)
    dj_db = 0.0

    for i in range(len(y)):
        error = predictions[i] - y[i]
        for j in range(len(weights)):
            # sum the values of the partial derivative with respect to w of the cost function with each prediction's error substituted in
            dj_dw[j] += error * X[j][i]

        dj_db += predictions[i] - y[i]

    dj_dw /= len(y)
    weights = weights - learning_rate * dj_dw
    print(weights, dj_dw)

    dj_db /= len(y)
    bias = bias - learning_rate * dj_db
    return weights, bias


def train_model(X, predictions, y, weights, bias=0.0, epochs=1, learning_rate=0.001):
    """
    Calculates new values of the model parameters using gradient descent by way of the derivative of the cost function

    Args:
      X (ndarray): Shape (n,m) array of m input vectors with n features
      predictions (ndarray): Shape (m,) array of predictions for y
      y (ndarray): Shape (m,) true values of y
      weights (ndarray): Shape (n,) model parameters
      bias (scalar):  model parameter
      learning_rate (scalar): the magnitude by which to modify the model parameters
      epochs (scalar): number of times to update the model parameters

    Returns:
      weights (ndarray): Shape (n,) final model parameters
      bias (scalar): final model parameter
    """
    for i in range(epochs):
        weights, bias = update_model_parameters(
            X, weights, bias, predictions, y, learning_rate
        )
        predictions = model(X, weights, bias)
        mse = mean_squared_error(predictions, y)

        print(f"Epoch {i}: weights = {weights}, bias = {bias}, MSE = {mse}")

    return weights, bias
