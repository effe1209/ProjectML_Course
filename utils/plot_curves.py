import numpy as np
import matplotlib.pyplot as plt
import os

def plot_curves(train_loss_vec: np.ndarray, val_loss_vec: np.ndarray, y_label: str = 'loss', val_str: str = 'val', title: str = '', save_plots: bool = False):
  """
  takes train_loss_vec and val_loss_vec and plots them
  """
  # Prettyer fonts
  plt.rc('font', size=14)
  plt.rc('axes', titlesize=16, labelsize=14, linewidth=1.5, grid=True)
  plt.rc('xtick', labelsize=14)
  plt.rc('ytick', labelsize=14)
  plt.rc('legend', fontsize=14)
  plt.rc('figure', titlesize=16)
  plt.rc('lines', linewidth=2)
  plt.rc('grid', alpha=0.6, linestyle='--')
  
  plt.figure(figsize=(5, 3))
  # Create the plot
  plt.plot(train_loss_vec, label = f'train {y_label}')
  plt.plot(val_loss_vec, label = f'{val_str} {y_label}')
  plt.xlabel(f'epoch')
  plt.ylabel(y_label)
  plt.legend()
  plt.title(title)

  #Save and show
  if save_plots:
    plt.savefig(f'plots/{title}.pdf', format='pdf')

  plt.show()