# ProjectML_Course
Machine learning - Project_A 

**Authors:** Alessandro Testi, Elia Bocini, Francesco Fiaschi
**Course:** Machine Learning
**Date:** 02-2026

## Project Aim
The aim of this project is to implemente a Neural Network framework built only with NumPy library.
The neural network is used to test and evaluate the performance on:
- *Monk's Dataset:* this is a standard benchmark for evaluation
- *ML Cup:* a dataset for regression task provided by course.

### Dataset Structure
#### Monk
**Variable Table**
| Variable            | Name    | Role    | Type | Description	Units | Missing Values |
| ------------------- | ------- | ------- | ---- | ----------------- | -------------- |
| class	Target	Binary |         |         |      |                   | no             |
| a1                  | Feature | Integer |      |                   | no             |
| a2                  | Feature | Integer |      |                   | no             |
| a3                  | Feature | Integer |      |                   | no             |
| a4                  | Feature | Integer |      |                   | no             |

**Feature Descriprion**
1. class: 0, 1 
2. a1:    1, 2, 3
3. a2:    1, 2, 3
4. a3:    1, 2
5. a4:    1, 2, 3
6. a5:    1, 2, 3, 4
7. a6:    1, 2
8. Id:    (A unique symbol for each instance)

## Implementation Details
**Training Set**
| ID  | Inputs [1-9] | Target_1 | Target_2 | Target_3 | Target_4 |
| --- | -------------- | -------- | -------- | -------- | -------- |
| 1   | Float          | Float    | FLoat    | Float    | Float    |

**Blind Test Set**
| ID  | Inputs [1-9] |
| --- | -------------- |
| 1   | Float          |

Un po' di note sparse, poi scriveremo benme e in inglese
i dataset devno essere in data con struttura

## Code Structure
```text
ROOT/
├── data/
│   ├── ML CUP/ roba per ml cup, per ora vuoto
│   └── monk/  cartella monk scaricata dal sito delle slide del prof, unzippata
│
├── model/
│   ├── activations/  tutte le funzioni di attivazione .py
│   ├── layers/
│   │   ├── layer.py  contiene la classe layer con proiezione lineare dei vettori e possibilita; di usare le funzioni di attivazione
│   │   └── random_layer.py continene una funzione per inizializzare i pesi del layer
│   ├── losses/
│   │   ├── (altre loss).py  tutte le funzioni di loss . py
│   │   └── loss.py classe per inizializzare la loss, calcola sia loss che la sua derivata
│   ├── network.py  contiene la classe Neural Network che concatena vari layer, implementa L2 e Momentum nella gradient descent
│   └── trainer.py  contiene una classe per automatizzare l'addestramento della rete neurale
│
├── utils/
│   ├── data_loader.pclasse che gestisce i dataset, puo' fare shuffle, train-val split, k-fold split, batch split
│   ├── monk_onehot.py   funzione per fare one hot encoding del monk, dato che i nunique del dataset sono noti
│   └── standard_scaler.py classe per fare standard scaling, fa fit con init, poi ha metodi transform e inverse transform
│
├── monk.ipynb    notebook con grid search e addestramento sui 3 monk problems
└── requirements.txt     contiene i requirements
```
