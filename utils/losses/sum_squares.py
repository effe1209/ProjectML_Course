import numpy as np

def sum_squares(pred : np.ndarray, actual: np.ndarray) -> np.ndarray:
    """
    Squared Error loss function, it is the one used in .pdf di Micheli "Backprop. notes latex".

    Args:
        pred (np.ndarray): predicted values
        actual (np.ndarray): actual values

    Returns:
        np.ndarray: sum squares loss of all samples of the batch

    Raises:
        Error: If the two arrays have different lenghts.
    """
    assert pred.shape == actual.shape, "Predicted vector and Actual vector have two different lenghts"

    return 0.5 * np.sum((pred - actual)**2, axis = 1)

def sum_squares_derivative(pred : np.ndarray, actual: np.ndarray) -> np.ndarray:
    """
    Squared Error derivarive.

    Args:
        pred (np.ndarray): predicted values
        actual (np.ndarray): actual values

    Returns:
        np.ndarray: Gradient Vector

    Raises:
        Error: If the two arrays have different lenghts.
    """
    assert pred.shape == actual.shape, "Predicted vector and Actual vector have two different lenghts"

    return pred - actual