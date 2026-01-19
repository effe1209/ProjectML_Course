import numpy as np

def leaky_relu(x : np.ndarray) -> np.ndarray:
    """
    Rectified leaku linear unit activation function.

    All values <= 0 are multiplied by 0.01.
    It shows better training properties than relu, relu derivative is 0 often
    impairing training

    Args:
        x (np.ndarray): input vector

    Returns:
        np.ndarray: Leaky Relu activated vector.
    """

    return np.where(x > 0, x, x * 0.01)

def leaky_relu_derivative(x : np.ndarray) -> np.ndarray:
    """
    Derivative of Leaky Rectified linear unit activation function.

    All values <= 0 have derivative 0.01, all values > 0 have derivative 1.

    Args:
        x (np.ndarray): input vector

    Returns:
        np.ndarray: gradient vector.
    """

    return np.where(x > 0, 1, 0.01)