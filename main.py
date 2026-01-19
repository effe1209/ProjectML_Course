import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

from nn import NeuralNetwork, Trainer
from utils.losses import Loss
from utils.data_manage import DataLoader, StandardScaler
from utils.activations.sigmoid import sigmoid

def get_monks_data(dataset_id=70):
    """
    Scarica e pre-processa il dataset MONK (One-Hot Encoding manuale incluso).
    """
    PATH_TRAIN = 'dataset/monks-1.train'
    PATH_TEST = 'dataset/monks-1.test'
    print(f"Scaricamento dataset ID {dataset_id}...")
    df_train = pd.read_csv(PATH_TRAIN, delim_whitespace=True, header=None)
    df_test = pd.read_csv(PATH_TEST, delim_whitespace=True, header=None)
    
    Train_set = np.array(df_train)
    Test_set = np.array(df_test)

    X_Train = Train_set[:, 1: -1].astype(int) # dalla prima alla penultima colonna, astype int usato se no np.eye non funziona
    y_Train = Train_set[:, 0].astype(int) # prima colonna
    y_Train = np.reshape(y_Train, (y_Train.shape[0], 1)) #(124,) -> (124,1)

    X_Test = Test_set[:, 1: -1].astype(int)
    y_Test = Test_set[:, 0].astype(int)
    y_Test = np.reshape(y_Test, (y_Test.shape[0], 1))

    X_Train.shape, y_Train.shape, X_Test.shape, y_Test.shape

    X_Train, X_Test = one_hot_encode(X_Train, X_Test)

    return X_Train, y_Train, X_Test, y_Test

def one_hot_encode(X_Train, X_Test):
    X_train_onehot = []
    X_dataset = np.concatenate((X_Train, X_Test), axis=0)
    print(X_dataset.shape)
    for column in range(X_dataset.shape[1]):
        X_dataset[:, column] = X_dataset[:, column] - 1 #needed to use np.eye
        n_unique = len(np.unique(X_dataset[:, column]))
        X_train_onehot.append(np.eye(n_unique)[X_dataset[:, column]])
    X_onehot = np.concatenate(X_train_onehot, axis=1)

    X_train = X_onehot[:X_Train.shape[0]]
    X_test = X_onehot[X_Train.shape[0]:]

    X_train.shape,  X_test.shape
    return X_train, X_test

def main():
    X_train, y_Train, X_test, y_Test = get_monks_data()

    monk_dataset = DataLoader(X_train, y_Train)
    K_FOLDS = monk_dataset.k_fold(k = 5)

    NEURAL_NETWORK_CONFIGURATIONS = [([17, 8, 8, 1],['tanh', 'tanh', 'identity'], 'binary cross entropy sigmoid'),
                                    ([17, 8, 8, 1],['leaky relu', 'leaky relu', 'sigmoid'], 'mse'),
                                    ]
    ETA_CONFIGURATIONS = [0.01, 0.001]
    LAMBDA_CONFIGURATIONS = [0, 1e-3]
    ALPHA_CONFIGURATIONS = [0, 0.5]
    EPOCHS = 300

    # (NEURAL_NETWORK_ARCHITECTURE, NEURAL_NETWORK_ACTIVATION, LOSS_F, ETA, LAMBDA, ALPHA)
    CONFIGURATIONS = []

    for NEURAL_NETWORK_ARCHITECTURE, NEURAL_NETWORK_ACTIVATION, LOSS_F in NEURAL_NETWORK_CONFIGURATIONS:
        for ETA in ETA_CONFIGURATIONS:
            for LAMBDA in LAMBDA_CONFIGURATIONS:
                for ALPHA in ALPHA_CONFIGURATIONS:
                    config = (NEURAL_NETWORK_ARCHITECTURE, NEURAL_NETWORK_ACTIVATION, LOSS_F, ETA, LAMBDA, ALPHA)
                    CONFIGURATIONS.append(config)

    CONFIG_DICTIONARY = {}

    for i in range(len(CONFIGURATIONS)):
        CONFIG_DICTIONARY[i] = 0

    for i in range(len(CONFIGURATIONS)):
        config = CONFIGURATIONS[i]
        for X_train, y_train, X_val, y_val in K_FOLDS:
            NEURAL_NETWORK_ARCHITECTURE, NEURAL_NETWORK_ACTIVATION, LOSS_F, ETA, LAMBDA, ALPHA = config
            print(f"NEURAL_NETWORK_ARCHITECTURE: {NEURAL_NETWORK_ARCHITECTURE}, NEURAL_NETWORK_ACTIVATION: {NEURAL_NETWORK_ACTIVATION}, LOSS_F: {LOSS_F}, ETA: {ETA}, LAMBDA: {LAMBDA}, ALPHA: {ALPHA}")
            #train
            nn = NeuralNetwork(NEURAL_NETWORK_ARCHITECTURE, NEURAL_NETWORK_ACTIVATION, std=0.2)
            trainer = Trainer(
                nn=nn,
                loss=Loss(LOSS_F),
                X_train=X_train,
                y_train=y_train, #no scaling y because of onehot
                X_val=X_val,
                y_val=y_val,
                epochs=EPOCHS,
                early_stopping=100, # no improvements in 50 epochs_> stop
                eta=ETA,               # Learning rate iniziale
                lam=LAMBDA,                # L2
                alpha=ALPHA,               # Momentum
                batch_size=16,
                shuffle_batches=True
            )
            # return_best_nn=True returns the best nn
            best_nn = trainer.train(min_clip=-1, max_clip=1, return_best_nn=True, print_epochs=False, plot_epochs=True)
            #best val accuracy
            out = best_nn.forward(X_val)[-1][-1]
            if LOSS_F == 'binary cross entropy sigmoid':
                out = sigmoid(out)

            predictions = np.round(out)
            print(f"Accuracy: {np.mean(predictions == y_val) * 100}%")
            print(CONFIG_DICTIONARY[i] )
            CONFIG_DICTIONARY[i] += np.mean(predictions == y_val)

if __name__ == "__main__":
    main()