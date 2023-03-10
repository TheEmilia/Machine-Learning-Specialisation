import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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

    predictions = sigmoid(np.dot(weights, X) + bias)

    return predictions


def compute_cost(predictions, y):
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


def compute_gradient(X, y, weights, bias):
    """
    Computes the gradient for logistic regression

    Args:
      X (ndarray (m,n): Data, m examples with n features
      y (ndarray (m,)): target values
      w (ndarray (n,)): model parameters
      b (scalar)      : model parameter
    Returns
      dj_dw (ndarray (n,)): The gradient of the cost w.r.t. the parameters w.
      dj_db (scalar)      : The gradient of the cost w.r.t. the parameter b.
    """

    dj_dw = np.zeros(weights.shape)
    dj_db = 0.0

    predictions = model(X, weights, bias)
    error = predictions - y

    for i in range(len(y)):
        for j in range(len(weights)):
            dj_dw[j] += error[i] * X[j][i]  # scalar
        dj_db += error[i]

    dj_dw /= len(y)  # (n,)
    dj_db /= len(y)  # scalar

    return dj_db, dj_dw


def gradient_descent(X, y, weights, bias=0.0, iterations=1, learning_rate=0.001):
    """
    Performs batch gradient descent

    Args:
      X (ndarray (n,m)): Data, m examples with n features
      y (ndarray (m,)): Target values
      weights (ndarray (n,)): Initial values of model parameters
      bias (scalar):  Initial values of model parameter
      iterations (scalar): number of times to perform gradient descent
      learning_rate (scalar): Learning rate

    Returns:
      weights (ndarray (n,)): Updated values of model parameters
      bias (scalar): Updated value of model parameter
    """

    for i in range(iterations):
        # Calculate the gradient and update the parameters
        dj_db, dj_dw = compute_gradient(X, y, weights, bias)

        # Update Parameters using w, b, learning_rate and gradient
        weights -= learning_rate * dj_dw
        bias -= learning_rate * dj_db

        predictions = model(X, weights, bias)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % np.ceil(iterations / 10) == 0:
            print(f"Iteration {i}: Cost {compute_cost(predictions, y)}")

    return weights, bias
