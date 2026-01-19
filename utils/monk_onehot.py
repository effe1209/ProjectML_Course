from __future__ import annotations
import numpy as np

def onehot_monk(x: np.ndarray) -> np.ndarray:
  '''
  Takes a monk not one hotted np.array and returns a one hotted one
  '''
  x_onehot = []
  columns_cardinality = [3, 3, 2, 3, 4, 2] #given by the problem
  for column_id in range(x.shape[1]):
    n_unique = columns_cardinality[column_id]
    x_onehot.append(np.eye(n_unique)[x[:, column_id] - 1])
  x_onehot = np.concatenate(x_onehot, axis=1)
  return x_onehot