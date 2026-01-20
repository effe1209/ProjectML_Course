# ProjectML_Course
Machine learning - Project_A 

**Authors:** Alessandro Testi, Elia Bocini, Francesco Fiaschi  
**Course:** Machine Learning  
**Date:** February 2026  

## Project Aim
The aim of this project is to implement a Neural Network framework built from scratch using only NumPy library.
The neural network framework is used to test and evaluate the performance on two different task:
- *Monk's Dataset:* a standard benchmark for classification task
- *ML Cup:* dataset for regression task provided by course.

## Implementation Details
The framework is divided in object class to ensure modularization, legibility and security.
### Architecture
- **Layers**: the network is composed of *fully connected layer*. Each layer is indipendent and managing:
  - **Weights and Biases**: to compute the linear projection (`net` value)
  - **Activation Function**: to produce the output the of the layer (non-linear)
  - **Backward Pass**: to propagate the error `delta` through the weights (*chain rule*)
- **Neural Network**: the main class that create the entire architecture.
  - 

## Code Structure
```text
ROOT/
├── data/
│   ├── ML CUP/                 # Dataset for ML Cup
│   └── monk/                   # Monk dataset (unzip folder)
│
├── model/
│   ├── activations/            # All file .py of activation functions
│   ├── layers/
│   │   ├── layer.py            # Class which implements activation functions and forward pass
│   │   └── random_layer.py     # Function to handle the random initialization
│   ├── losses/
│   │   ├── ...                 # All file .py of loss functions (e.g. MSE, CrossEntropy)
│   │   └── loss.py             # Class which computes the loss and its derivative
│   ├── network.py              # Neural Network class supporting multi-layer, L2 regularization and momentum
│   └── trainer.py              # Automatically training process for neural network
│
├── utils/
│   ├── data_loader.py          # Handles dataset, including: shuffling, validation split, k-fold and mini-batch
│   ├── monk_onehot.py          # Applies one hot encoding to Monk dataset (known cardinality)
│   └── standard_scaler.py      # Standard scaler to perform Standardization and inverse transformation
│
├── monk.ipynb                  # Experiment notebook: model selection and model assessment with grid search for Monk
├── mlcup.ipynb.                # Experiment notebook for ML Cup with final result
└── requirements.txt            # Project dependencies - Required to run the code
```

## Installation
To execute the experiments, ensure you have **Python** installed.
We strongly recommended to set up a virtual enviroment to exclude issuess on dependencies.
To install the required packages run the following command:

```bash
pip install -r requirements.txt
```

## Usage
### Reproducing Experiment
To reproduce the experiment on your device and to see the result presented in the report:
1. Open the jupyter file (`monk.ipynb` or `mlcup.ipynb`)
2. Run all cells sequentially.
### Customization
The notebook by default make a grid search to tune the hyperparameters.
You can costomization the search space of hyperparameter by modify `CONFIGURATIONS` dictionary.
- **Log:** To print the the loss at each epoch: `print_epochs=False` in `trainer.train()`
- **Plot:** To generate and print the plot of loss curves: `plot_epochs=True` in `trainer.train()`
    **!Warning**: enabling the print of plot during grid search will slow down the execution and the output contains too many graphs.

### Dataset Structure
#### Monk
Structure of the dataset for classification task.
One-Hot Encoding: features cardinality is known
**Variable Table**
| Variable Name | Role    | Type        | Description | Units | Missing Values |
| ------------- | ------- | ----------- | ----------- | ----- | -------------- |
| class         | Target  | Binary      |             |       | no             |
| a1            | Feature | Integer     |             |       | no             |
| a2            | Feature | Integer     |             |       | no             |
| a3            | Feature | Integer     |             |       | no             |
| a4            | Feature | Integer     |             |       | no             |
| ID            | ID      | Categorical |             |       | no             |

**Feature Descriprion**
1. class: 0, 1 
2. a1:    1, 2, 3
3. a2:    1, 2, 3
4. a3:    1, 2
5. a4:    1, 2, 3
6. a5:    1, 2, 3, 4
7. a6:    1, 2
8. Id:    (A unique symbol for each instance)

### Ml_Cup
**Training Set**
| ID  | Inputs [2-9] | Target_1 | Target_2 | Target_3 | Target_4 |
| --- | ------------ | -------- | -------- | -------- | -------- |
| 1   | Float        | Float    | FLoat    | Float    | Float    |

**Blind Test Set**
| ID  | Inputs [2-9] |
| --- | ------------ |
| 1   | Float        |
