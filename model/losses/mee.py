import numpy as np

def mee(pred : np.ndarray, actual: np.ndarray) -> np.ndarray:
    """
    Mean Euclidean Error loss function.

    Args:
        pred (np.ndarray): predicted values
        actual (np.ndarray): actual values

    Returns:
        np.ndarray: MEE loss of all samples of the batch

    Raises:
        Error: If the two arrays have different lenghts.
    """
    assert pred.shape == actual.shape, "Predicted vector and Actual vector have two different lenghts"

    return np.sqrt(np.sum((pred - actual)**2, axis = 1))