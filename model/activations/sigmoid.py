import numpy as np

def sigmoid(x : np.ndarray) -> np.ndarray:
    """
    All entries of the vector are sigmoid activated.

    Args:
        x (np.ndarray): input vector

    Returns:
        np.ndarray: Sigmoid activated vector.
    """
    return 1/(1 + np.exp(-x))

def sigmoid_derivative(x : np.ndarray) -> np.ndarray:
    """
    Derivative of sigmoid activation function.

    Args:
        x (np.ndarray): input vector

    Returns:
        np.ndarray: gradient of Sigmoid activated vector.
    """
    return sigmoid(x) * (1 - sigmoid(x))