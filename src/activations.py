from __future__ import annotations
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

def softmax(x : np.ndarray) -> np.ndarray:
    """
    Softmax activation of the vector.

    Args:
        x (np.ndarray): input vector

    Returns:
        np.ndarray: Softmax activated vector.
    """
    e = np.exp(x)
    return (e/np.sum(e, axis = 1, keepdims = True)) #axis = 1 outputs a vector with n_batch dims, keepdims needed for broadcasting makes it (1, n_batch)