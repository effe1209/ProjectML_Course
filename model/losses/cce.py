import numpy as np
from model.activations import softmax

def cross_entropy_loss(pred : np.ndarray, actual: np.ndarray) -> np.ndarray:
    """
    Cross Entropy loss function.

    Args:
        pred (np.ndarray): predicted classes
        actual (np.ndarray): actual classe

    Returns:
        np.ndarray: CE loss vec

    Raises:
        Error: If the two arrays have different lenghts.
    """
    assert pred.shape == actual.shape, "Predicted vector and Actual vector have two different lenghts"

    return np.sum(- actual * np.log(pred + 1e-15), axis = 1)

def softmax_cross_entropy_loss(pred : np.ndarray, actual: np.ndarray) -> np.ndarray:
  """
  Softmax activates the input vector then Cross Entropy loss.

  Args:
      pred (np.ndarray): predicted classes (pre softmax)
      actual (np.ndarray): actual class
  """
  assert pred.shape == actual.shape, "Predicted vector and Actual vector have two different lenghts"

  return cross_entropy_loss(softmax(pred), actual)

def softmax_cross_entropy_derivative(pred : np.ndarray, actual: np.ndarray) -> np.ndarray:
  """
  Derviative of softmax then cross E.
    Args:
      pred (np.ndarray): predicted classes (pre - softmax)
      actual (np.ndarray): actual class
  """

  assert pred.shape == actual.shape, "Predicted vector and Actual vector have two different lenghts"
  return softmax(pred) - actual