import numpy as np

def mse(pred : np.ndarray, actual: np.ndarray) -> np.ndarray:
    """
    Mean Squared Error loss function.

    Args:
        pred (np.ndarray): predicted values
        actual (np.ndarray): actual values

    Returns:
        np.ndarray: MSE loss of all samples of the batch

    Raises:
        Error: If the two arrays have different lenghts.
    """
    assert pred.shape == actual.shape, "Predicted vector and Actual vector have two different lenghts"

    return np.mean((pred - actual)**2, axis = 1)

def mse_derivative(pred : np.ndarray, actual: np.ndarray) -> np.ndarray:
    """
    Mean Squared Error derivarive.

    Args:
        pred (np.ndarray): predicted values
        actual (np.ndarray): actual values

    Returns:
        np.ndarray: Gradient Vector

    Raises:
        Error: If the two arrays have different lenghts.
    """
    assert pred.shape == actual.shape, "Predicted vector and Actual vector have two different lenghts"

    return 2 * (pred - actual) / pred.shape[1]