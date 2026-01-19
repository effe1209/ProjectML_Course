import numpy as np

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

