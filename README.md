# ProjectML_Course
Machine learning - Project_A 

Un po' di note sparse, poi scriveremo benme e in inglese
i dataset devno essere in data con struttura

```text
PROJECT_ROOT/
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