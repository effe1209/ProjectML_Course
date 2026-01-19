import numpy as np

class StandardScaler:
  """
  Class for preprocessing data
  """
  def __init__(self, data: np.ndarray):
    assert data.ndim == 2, "I need a 2d matrix for preprocessing"
    self.row_mean = np.mean(data, axis = 0)
    self.row_std = np.std(data, axis = 0)

  def transform(self, data: np.ndarray) -> np.ndarray:
    """
    Transforms the data accordingly
    """
    assert data.ndim == 2, "I need a 2d matrix for preprocessing"
    assert 0. not in self.row_std, "Can't divide by 0 in std scaling"
    return (data - self.row_mean) / self.row_std

  def inverse_transform(self, data: np.ndarray) -> np.ndarray:
    """
    Inverse Transforms the data accordingly
    """
    assert data.ndim == 2, "I need a 2d matrix for preprocessing"
    return data * self.row_std + self.row_mean