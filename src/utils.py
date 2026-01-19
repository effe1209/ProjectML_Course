from __future__ import annotations
import numpy as np
import copy

class DataLoader:
  """
  Manages the datasets
  """
  def __init__(self, X_dataset: np.ndarray, y_dataset: np.ndarray):
    """
    Arhgs:
      X_dataset (np.ndarray): X dataset
      y_dataset (np.ndarray): y dataset
    """
    self.X_train, self.y_train = X_dataset, y_dataset

  def shuffle(self, shuffle: bool, X: np.ndarray, y:np.ndarray):
    """
    Shuffle the dataset
    Args:
      shuffle (bool): if true shuffle the dataset, return dataset else
      X (np.ndarray): X dataset
      y (np.ndarray): y dataset
    """
    if shuffle:
      length_dataset = len(X)
      shuffled_indices =  np.arange(length_dataset)
      np.random.shuffle(shuffled_indices)
      return (X[shuffled_indices], y[shuffled_indices])
    else:
      return (X, y)

  def train_val_split(self, portion: float = 0.8, shuffle: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split dataset in train and validation set, keeping proportions
    Args:
      portion (float, optional): portion of the train dataset
      shuffle (bool, optional): shuffles the dataset if true
    """
    length_dataset = len(self.X_train)
    X, y = self.shuffle(shuffle, self.X_train, self.y_train)

    X_train = X[: int(length_dataset * portion)]
    y_train = y[: int(length_dataset * portion)]
    X_val = X[int(length_dataset * portion):]
    y_val = y[int(length_dataset * portion):]

    return X_train, y_train, X_val, y_val


  def k_fold_split(self, k: int = 4, shuffle: bool = True) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Splits the dataset un k folds, returns a list of tuples with the splits
    Args:
      k (int, optional): number of folds
      shuffle (bool, optional): shuffles the dataset if true  
   
    """
    length_dataset = len(self.X_train)
    datasets = [] #list of tuples of split datasets

    # creates a list of points where the dataset has to be cut
    cut_indices = [round(length_dataset * (x)/ k) for x in range(k + 1)]

    X, y = self.shuffle(shuffle, self.X_train, self.y_train)

    for cut in range(len(cut_indices) - 1):
      X_train = X[cut_indices[cut]: cut_indices[cut + 1]]
      y_train = y[cut_indices[cut]: cut_indices[cut + 1]]
      datasets.append((X_train, y_train))

    return datasets

  def k_fold(self,  k: int = 4, shuffle: bool = True) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    returns a list of k tuples each one containing train x,y and val x,y
    Args:
      k (int, optional): number of folds
      shuffle (bool, optional): shuffles the dataset if true
    """
    datasets = self.k_fold_split(k, shuffle)
    k_split_datasets = []
    for i in range(len(datasets)):
      data = copy.deepcopy(datasets)
      val = data.pop(i)
      X_val = val[0]
      y_val = val[1]
      X_train = np.concatenate([x[0] for x in data])
      y_train = np.concatenate([x[1] for x in data])
      k_split_datasets.append((X_train, y_train, X_val, y_val))

    return k_split_datasets

  def get_batches(self, dataset : tuple[np.ndarray, np.ndarray], batch_size : int = 32, shuffle: bool = True, keep_last : bool = False) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Splits the dataset in batches
    Args:
      dataset (tuple[np.ndarray, np.ndarray]): dataset to split
      batch_size (int, optional): batch size
      shuffle (bool, optional): shuffle?
      keep_last (bool, optional): if the last batch is smaller than batch size, keep it?
    """
    length_dataset = len(dataset[0])
    batches = []

    #shuffle
    X, y = self.shuffle(shuffle, dataset[0], dataset[1])

    # creates a list of points where the dataset has to be cut
    cut_indices = []
    for _ in range(length_dataset // batch_size + 1):
      cut_indices.append(_ * batch_size)

    #we keep the remaining part?
    if keep_last and cut_indices[-1] < length_dataset: #cant add 2 times the same last value
      cut_indices.append(length_dataset)

    for cut in range(len(cut_indices) - 1):
      X_train = X[cut_indices[cut]: cut_indices[cut + 1]]
      y_train = y[cut_indices[cut]: cut_indices[cut + 1]]
      batches.append((X_train, y_train))

    return batches

class StandardScaler:
  """
  Class for preprocessing data
  """
  def __init__(self, data: np.ndarray):
    """
    Args:
      data (np.ndarray): data to preprocess
    """
    assert data.ndim == 2, "I need a 2d matrix for preprocessing"
    self.row_mean = np.mean(data, axis = 0)
    self.row_std = np.std(data, axis = 0)

  def transform(self, data: np.ndarray) -> np.ndarray:
    """
    Transforms the data accordingly
    Args:
      data (np.ndarray): data to preprocess
    """
    assert data.ndim == 2, "I need a 2d matrix for preprocessing"
    assert 0. not in self.row_std, "Can't divide by 0 in std scaling"
    return (data - self.row_mean) / self.row_std

  def inverse_transform(self, data: np.ndarray) -> np.ndarray:
    """
    Inverse Transforms the data accordingly
    Args:
      data (np.ndarray): data to preprocess
    """
    assert data.ndim == 2, "I need a 2d matrix for preprocessing"
    return data * self.row_std + self.row_mean


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