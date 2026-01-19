import numpy as np
from utils.activations import sigmoid

def binary_cross_entropy_loss(pred : np.ndarray, actual: np.ndarray) -> np.ndarray:
    """
    Binary Cross Entropy loss function.

    Args:
        pred (np.ndarray): predicted classes
        actual (np.ndarray): actual classe

    Returns:
        np.ndarray: CE loss vec

    Raises:
        Error: If the two arrays have different lenghts.
    """
    assert pred.shape == actual.shape, "Predicted vector and Actual vector have two different lenghts"
    assert pred.ndim == 2, "Pred must be a vector with shape (batch_size, 1)"
    assert pred.shape[1] == 1, "Pred must be a vector with shape (batch_size, 1)"

    return - actual * np.log(pred + 1e-15) - (1 - actual) * np.log(1 - pred + 1e-15)

def sigmoid_binary_cross_entropy_loss(pred : np.ndarray, actual: np.ndarray) -> np.ndarray:
  """
  Sigmoid activates the input vector then B. Cross Entropy loss.

  Args:
      pred (np.ndarray): predicted classes (pre sigmoid)
      actual (np.ndarray): actual class
  """
  assert pred.shape == actual.shape, "Predicted vector and Actual vector have two different lenghts"

  return binary_cross_entropy_loss(sigmoid(pred), actual)

def sigmoid_binary_cross_entropy_loss_derivative(pred : np.ndarray, actual: np.ndarray) -> np.ndarray:
  """
  Derivative of Sigmoid activates the input vector then B. Cross Entropy loss.

  Args:
      pred (np.ndarray): predicted classes (pre sigmoid)
      actual (np.ndarray): actual class
  """
  assert pred.shape == actual.shape, "Predicted vector and Actual vector have two different lenghts"

  return sigmoid(pred) - actual