import numpy as np

def tanh(x : np.ndarray) -> np.ndarray:
    """
    All entries of the vector are tanh activated.

    Args:
        x (np.ndarray): input vector

    Returns:
        np.ndarray: Tanh activated vector.
    """
    return np.tanh(x)

def tanh_derivative(x : np.ndarray) -> np.ndarray:
    """
    Derivative of tanh activation function.

    Args:
        x (np.ndarray): input vector

    Returns:
        np.ndarray: gradient of tanh activated vector.
    """
    return 1 / (np.cosh(x))**2