from __future__ import annotations
import pandas as pd
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


def load_monk(path_train: str, path_test: str):
    """
    Load the MONK dataset from the given file paths.
    Args:
    path_train (str): train path
    path_test (str): test path

    Returns:
    (np.ndarray): onehotencode train
    (np.ndarray): tran laberls
    (np.ndarray): onehot encode test
    (np.ndarray): test labels
    """
    df_train = pd.read_csv(path_train, sep='\s+', header=None) #sep='\s+' == delim white space
    df_test = pd.read_csv(path_test, sep='\s+', header=None)

    #Convert df in numpy arrays, excluding last column
    Train_set = np.array(df_train.iloc[:, :-1], dtype=np.int32)
    Test_set = np.array(df_test.iloc[:, :-1], dtype=np.int32)
    X_train_full = onehot_monk(Train_set[:, 1:]) 
    y_train_full = np.reshape(Train_set[:, 0], (len(Train_set), 1))# only first column, need shape (n,1) to work

    X_test = onehot_monk(Test_set[:, 1:]) 
    y_test = np.reshape(Test_set[:, 0], (len(Test_set), 1))
    print(f'After one-hot encoding: X train full shape: {X_train_full.shape}, X test shape: {X_test.shape}, y train full shape: {y_train_full.shape}, y test shape: {y_test.shape}')
    return X_train_full, y_train_full, X_test, y_test