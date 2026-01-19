import numpy as np

def relu(x : np.ndarray) -> np.ndarray:
    """
    Rectified linear unit activation function.

    All values <= 0 are set to 0.

    Args:
        x (np.ndarray): input vector

    Returns:
        np.ndarray: Relu activated vector.
    """

    return np.maximum(x, 0)

def relu_derivative(x : np.ndarray) -> np.ndarray:
    """
    Derivative of Rectified linear unit activation function.

    All values <= 0 have derivative 0, all values > 0 have derivative 1.

    Args:
        x (np.ndarray): input vector

    Returns:
        np.ndarray: gradient vector.
    """

    return np.where(x > 0, 1, 0)