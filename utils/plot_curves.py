import numpy as np
import matplotlib.pyplot as plt


def plot_loss(train_loss_vec: np.ndarray, val_loss_vec: np.ndarray, y_label: str = 'loss', val_str: str = 'val'):
  """
  takes train_loss_vec and val_loss_vec and plots them
  """
  plt.plot(train_loss_vec, label = f'train {y_label}')
  plt.plot(val_loss_vec, label = f'{val_str} {y_label}')
  plt.xlabel(f'epoch')
  plt.ylabel(y_label)
  plt.legend()
  plt.show()