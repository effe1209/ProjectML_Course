import numpy as np

from .mse import mse, mse_derivative
from .sum_squares import sum_squares, sum_squares_derivative
from .cce import softmax_cross_entropy_loss, softmax_cross_entropy_derivative
from .bce import sigmoid_binary_cross_entropy_loss, sigmoid_binary_cross_entropy_loss_derivative

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