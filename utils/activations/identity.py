import numpy as np

def identity(x : np.ndarray) -> np.ndarray:
    """
    Identity activation function.

    Args:
        x (np.ndarray): input vector

    Returns:
        np.ndarray: same vector.
    """

    return x

def identity_derivative(x : np.ndarray) -> np.ndarray:
    """
    Identity activation functionderivative.
    Simply returns a vector of ones.

    Args:
        x (np.ndarray): input vector

    Returns:
        np.ndarray: same vector shape of ones.
    """

    return np.ones(x.shape)