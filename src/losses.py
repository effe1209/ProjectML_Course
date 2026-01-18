from __future__ import annotations
from src.activations import sigmoid, softmax
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

class Loss:
  """
  Una classe per la loss che si inizializza con il nome della loss e contiene metodi per il calcolo di loss e gradiente
  """
  loss_dic = {'mse': mse, 'sum squares': sum_squares, 'cross entropy softmax': softmax_cross_entropy_loss, 'binary cross entropy sigmoid' : sigmoid_binary_cross_entropy_loss}
  loss_derivative_dic = {'mse': mse_derivative, 'sum squares': sum_squares_derivative, 'cross entropy softmax': softmax_cross_entropy_derivative, 'binary cross entropy sigmoid' : sigmoid_binary_cross_entropy_loss_derivative}

  def __init__(self, loss_f: str):
    """
    Inizializza la classe Loss
    Args:
      loss_f (str): nome della loss
    """
    assert loss_f in ['mse', 'sum squares', 'cross entropy softmax', 'binary cross entropy sigmoid'], "Loss not supported"
    self.loss_f = loss_f

  def compute_loss(self, pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """
    Calcola la loss del batch
    Args:
      pred (np.ndarray): out finale della nn
      actual (np.ndarray): batch di y
    Returns:
      np.ndarray: loss del batch
    """
    return self.loss_dic[self.loss_f](pred, actual)

  def compute_loss_gradient(self, pred: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """
    Calcola la loss del batch
    Args:
      pred (np.ndarray): out finale della nn
      actual (np.ndarray): batch di y
    Returns:
      np.ndarray: loss del batch
    """
    return self.loss_derivative_dic[self.loss_f](pred, actual)