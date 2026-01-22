import numpy as np
from utils.model_selection_helpers import instability_coeff, tran_val_diff
from utils.standard_scaler import StandardScaler
from model.trainer import Trainer
from model.network import NeuralNetwork
from model.losses import Loss
from model.losses import mee
from model.activations import sigmoid

def grid_search_mlcup(LEN_CONFIGURATIONS: int, CONFIGURATIONS: list, k_fold: list, EPOCHS: int, EARLY_STOPPING_PATIENCE: int):
    # Here we create dictionaries for storing avg accuracies and epochs on k folds and other metrics, we will use them to select the best configuration
    CONFIG_DICTIONARY, CONFIG_DICTIONARY_EPOCHS, CONFIG_DICTIONARY_INSTABILITY_TRAIN, CONFIG_DICTIONARY_INSTABILITY_VAL, CONFIG_DICTIONARY_TRAIN_LOSS_DIFF, CONFIG_DICTIONARY_TEST_LOSS, CONFIG_DICTIONARY_TRAIN_LOSS= {}, {}, {}, {}, {}, {}, {}
    for i in range(LEN_CONFIGURATIONS):
        CONFIG_DICTIONARY[i], CONFIG_DICTIONARY_EPOCHS[i], CONFIG_DICTIONARY_INSTABILITY_TRAIN[i], CONFIG_DICTIONARY_INSTABILITY_VAL[i], CONFIG_DICTIONARY_TRAIN_LOSS_DIFF[i], CONFIG_DICTIONARY_TEST_LOSS[i], CONFIG_DICTIONARY_TRAIN_LOSS[i] = 0, 0, 0, 0, 0, [], []
    
    # Cross Validation X K-folds
    for i in range(LEN_CONFIGURATIONS): #iterate over all configurations
        print(f"Training {i+1}/{LEN_CONFIGURATIONS}")
        config = CONFIGURATIONS[i]
        
        for X_t, y_t, X_v, y_v in k_fold: #iterate over k folds
            NEURAL_NETWORK_ARCHITECTURE, NEURAL_NETWORK_ACTIVATION, LOSS_F, ETA, LAMBDA, ALPHA, BATCH_SIZE = config
            print(f"NEURAL_NETWORK_ARCHITECTURE: {NEURAL_NETWORK_ARCHITECTURE}, NEURAL_NETWORK_ACTIVATION: {NEURAL_NETWORK_ACTIVATION}, LOSS_F: {LOSS_F}, ETA: {ETA}, LAMBDA: {LAMBDA}, ALPHA: {ALPHA}, BATCH_SIZE: {BATCH_SIZE}")
            #scaling
            X_scaler = StandardScaler(X_t)
            X_train_scaled = X_scaler.transform(X_t)
            X_val_scaled = X_scaler.transform(X_v)
            y_scaler = StandardScaler(y_t)
            y_train_scaled = y_scaler.transform(y_t)
            y_val_scaled = y_scaler.transform(y_v)
            #train
            nn = NeuralNetwork(NEURAL_NETWORK_ARCHITECTURE, NEURAL_NETWORK_ACTIVATION)
            trainer = Trainer(
                nn=nn,
                loss=Loss(LOSS_F),
                X_train=X_train_scaled,
                y_train=y_train_scaled, 
                X_val=X_val_scaled,
                y_val=y_val_scaled,
                epochs=EPOCHS,
                early_stopping=EARLY_STOPPING_PATIENCE, # no improvements in  epochs-> stop
                min_improvement=0.0,       #even the smallest improvement over val set is considered
                eta=ETA,                   # Learning rate iniziale
                lam=LAMBDA,                # L2
                alpha=ALPHA,               # Momentum
                batch_size=BATCH_SIZE,
                shuffle_batches=True
            )
            best_nn, train_loss_vector, test_loss_vector = trainer.train(print_epochs=False) # returns the best nn based on val, the train and val loss vectors, best nn not used otherwise leakage
            CONFIG_DICTIONARY_EPOCHS[i] += len(train_loss_vector) #number of epochs until early stopping or max epochs
            CONFIG_DICTIONARY_INSTABILITY_TRAIN[i] += instability_coeff(train_loss_vector)
            CONFIG_DICTIONARY_INSTABILITY_VAL[i] += instability_coeff(test_loss_vector)
            CONFIG_DICTIONARY_TRAIN_LOSS_DIFF[i] += tran_val_diff(train_loss_vector, test_loss_vector)
            # nn val loss and std
            out = nn.forward(X_val_scaled)[-1][-1]
            out = y_scaler.inverse_transform(out)
            CONFIG_DICTIONARY_TEST_LOSS[i].append( np.mean(mee(out, y_v)))
            # nn val loss and std
            out = nn.forward(X_train_scaled)[-1][-1]
            out = y_scaler.inverse_transform(out)
            CONFIG_DICTIONARY_TRAIN_LOSS[i].append( np.mean(mee(out, y_t)))
            #val accuracy
            out = best_nn.forward(X_val_scaled)[-1][-1]
            out = y_scaler.inverse_transform(out)
            print(f"Mee best: {np.mean(mee(out, y_v))}")
            CONFIG_DICTIONARY[i] += np.mean(mee(out, y_v))
    return CONFIG_DICTIONARY, CONFIG_DICTIONARY_EPOCHS, CONFIG_DICTIONARY_INSTABILITY_TRAIN, CONFIG_DICTIONARY_INSTABILITY_VAL, CONFIG_DICTIONARY_TRAIN_LOSS_DIFF, CONFIG_DICTIONARY_TEST_LOSS, CONFIG_DICTIONARY_TRAIN_LOSS


def grid_search_monk(LEN_CONFIGURATIONS: int, CONFIGURATIONS: list, k_fold: list, EPOCHS: int, EARLY_STOPPING_PATIENCE: int):
  # Here we create dictionaries for storing avg accuracies and epochs on k folds and other metrics, we will use them to select the best configuration
  CONFIG_DICTIONARY, CONFIG_DICTIONARY_EPOCHS, CONFIG_DICTIONARY_INSTABILITY_TRAIN, CONFIG_DICTIONARY_INSTABILITY_VAL, CONFIG_DICTIONARY_TRAIN_LOSS_DIFF = {}, {}, {}, {}, {}
  for i in range(LEN_CONFIGURATIONS):
    CONFIG_DICTIONARY[i], CONFIG_DICTIONARY_EPOCHS[i], CONFIG_DICTIONARY_INSTABILITY_TRAIN[i], CONFIG_DICTIONARY_INSTABILITY_VAL[i], CONFIG_DICTIONARY_TRAIN_LOSS_DIFF[i] = 0, 0, 0, 0, 0

  # Cross Validation X K-folds
  for i in range(LEN_CONFIGURATIONS): #iterate over all configurations
    print(f"Training {i+1}/{LEN_CONFIGURATIONS}")
    config = CONFIGURATIONS[i]
    
    for X_train, y_train, X_val, y_val in k_fold: #iterate over k folds
      NEURAL_NETWORK_ARCHITECTURE, NEURAL_NETWORK_ACTIVATION, LOSS_F, ETA, LAMBDA, ALPHA, BATCH_SIZE = config
      print(f"NEURAL_NETWORK_ARCHITECTURE: {NEURAL_NETWORK_ARCHITECTURE}, NEURAL_NETWORK_ACTIVATION: {NEURAL_NETWORK_ACTIVATION}, LOSS_F: {LOSS_F}, ETA: {ETA}, LAMBDA: {LAMBDA}, ALPHA: {ALPHA}, BATCH_SIZE: {BATCH_SIZE}")
      
      #train
      nn = NeuralNetwork(NEURAL_NETWORK_ARCHITECTURE, NEURAL_NETWORK_ACTIVATION)
      trainer = Trainer(
          nn=nn,
          loss=Loss(LOSS_F),
          X_train=X_train,
          y_train=y_train, 
          X_val=X_val,
          y_val=y_val,
          epochs=EPOCHS,
          early_stopping=EARLY_STOPPING_PATIENCE, # no improvements (implicitly greater than 5%) in 50 epochs-> stop
          eta=ETA,                   # Learning rate iniziale
          lam=LAMBDA,                # L2
          alpha=ALPHA,               # Momentum
          batch_size=BATCH_SIZE,
          shuffle_batches=True
      )
      _ , train_loss_vector, test_loss_vector = trainer.train(print_epochs=False) # returns the best nn based on val, the train and val loss vectors
      CONFIG_DICTIONARY_EPOCHS[i] += len(train_loss_vector) #number of epochs until early stopping or max epochs
      CONFIG_DICTIONARY_INSTABILITY_TRAIN[i] += instability_coeff(train_loss_vector)
      CONFIG_DICTIONARY_INSTABILITY_VAL[i] += instability_coeff(test_loss_vector)
      CONFIG_DICTIONARY_TRAIN_LOSS_DIFF[i] += tran_val_diff(train_loss_vector, test_loss_vector)
      #best val accuracy
      out = nn.forward(X_val)[-1][-1]
      if LOSS_F == 'binary cross entropy sigmoid':
        out = sigmoid(out)
      predictions = np.round(out)
      print(f"Accuracy: {np.mean(predictions == y_val) * 100}%")
      CONFIG_DICTIONARY[i] += np.mean(predictions == y_val)
  return CONFIG_DICTIONARY, CONFIG_DICTIONARY_EPOCHS, CONFIG_DICTIONARY_INSTABILITY_TRAIN, CONFIG_DICTIONARY_INSTABILITY_VAL, CONFIG_DICTIONARY_TRAIN_LOSS_DIFF