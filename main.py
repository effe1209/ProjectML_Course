import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from nn import NeuralNetwork, Trainer
from utils.losses import Loss
from utils.data_manage import DataLoader, StandardScaler

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo # Assicurati di aver fatto pip install ucimlrepo

# Import dai tuoi moduli (struttura core/utils)
from core.network import NeuralNetwork
from core.trainer import Trainer
from utils.losses import Loss
from utils.data_manage import DataLoader

def get_monks_data(dataset_id=70):
    """
    Scarica e pre-processa il dataset MONK (One-Hot Encoding manuale incluso).
    """
    print(f"Scaricamento dataset ID {dataset_id}...")
    monk_data = fetch_ucirepo(id=dataset_id)
    
    X = np.array(monk_data.data.features)
    y = np.array(monk_data.data.targets)
    
    # --- IL TUO CODICE DI ONE-HOT ENCODING ---
    # Nota: Assicura che X sia float/int per le operazioni matematiche
    X = X.astype(int) 
    
    X_onehot = []
    for column in range(X.shape[1]):
        # Shift a 0-based index se necessario (Monks usa 1,2,3...)
        # Attenzione: controlla se i valori partono da 1. Se sì, -1 è corretto.
        vals = X[:, column]
        if np.min(vals) > 0:
             vals = vals - 1
             
        n_unique = len(np.unique(vals))
        # Crea la matrice identità e seleziona le righe
        X_onehot.append(np.eye(n_unique)[vals])
        
    X_onehot = np.concatenate(X_onehot, axis=1)
    
    return X_onehot, y

def main():
    # 1. CARICAMENTO DATI
    X, y = get_monks_data(id=70) # Monks-1 o Monk-2
    
    # Split manuale o tramite DataLoader
    # Qui usiamo il DataLoader solo per lo split iniziale se vuoi, 
    # oppure passi tutto al Trainer se gestisce lui la validation.
    # Usiamo il tuo approccio:
    monk_dataset = DataLoader(X_dataset=X, y_dataset=y)
    X_train, y_train, X_val, y_val = monk_dataset.train_val_split(portion=0.75)
    
    print(f"Data Loaded. Train shape: {X_train.shape}, Val shape: {X_val.shape}")

    # 2. CONFIGURAZIONI (GRID SEARCH)
    NEURAL_NETWORK_CONFIGURATIONS = [
        ([17, 4, 1], ['tanh', 'sigmoid']), # Ridotto per test veloce, rimetti i tuoi
        ([17, 8, 1], ['relu', 'sigmoid']),
        # ... Incolla qui tutta la tua lista ...
    ]
    
    ETA_CONFIGURATIONS = [0.01, 0.1] # Metti i tuoi
    LAMBDA_CONFIGURATIONS = [0, 1e-4]
    ALPHA_CONFIGURATIONS = [0.0, 0.6]
    EPOCHS = 300
    
    # Lista per salvare i risultati
    results_data = []
    
    loss_fn = Loss('mse')

    # Contatore per log
    total_configs = len(NEURAL_NETWORK_CONFIGURATIONS) * len(ETA_CONFIGURATIONS) * len(LAMBDA_CONFIGURATIONS) * len(ALPHA_CONFIGURATIONS)
    current_config = 0

    # 3. CICLO DI TRAINING
    for arch, act in NEURAL_NETWORK_CONFIGURATIONS:
        for eta in ETA_CONFIGURATIONS:
            for lam in LAMBDA_CONFIGURATIONS:
                for alpha in ALPHA_CONFIGURATIONS:
                    current_config += 1
                    config_name = f"Arch:{arch}-Eta:{eta}-L2:{lam}-Alpha:{alpha}"
                    print(f"[{current_config}/{total_configs}] Running: {config_name}")

                    # Inizializza
                    nn = NeuralNetwork(arch, act, std=0.2)
                    
                    trainer = Trainer(
                        nn=nn,
                        loss=loss_fn,
                        X_train=X_train, y_train=y_train,
                        X_val=X_val, y_val=y_val,
                        epochs=EPOCHS,
                        early_stopping=50,
                        eta=eta,
                        lam=lam,
                        alpha=alpha,
                        tau=150,
                        std_scale_x=False, # Già one-hot
                        std_scale_y=False,
                        batch_size=16,
                        shuffle_batches=True
                    )

                    # --- ATTENZIONE: plot_epochs=False ---
                    # Altrimenti generi 200 finestre grafiche e blocchi tutto!
                    # Supponiamo che tu abbia modificato train() per restituire (best_nn, history)
                    # Se non l'hai fatto, trainer.train restituisce solo best_nn (come nel tuo codice originale)
                    try:
                        res = trainer.train(return_best_nn=True, print_epochs=False, plot_epochs=False)
                        
                        # Gestione ritorno (se hai modificato train o no)
                        if isinstance(res, tuple):
                            best_nn, history = res
                            val_acc = history['val_loss'][-1] # Esempio placeholder
                            epochs_run = history['epochs_run']
                        else:
                            best_nn = res
                            # Se trainer non restituisce history, calcoliamo accuracy manuale ORA
                            out = best_nn.forward(X_val)[-1][-1]
                            predictions = np.round(out)
                            val_acc = np.mean(predictions == y_val) * 100
                            epochs_run = "N/A"

                        print(f"   -> Accuracy: {val_acc:.2f}%")
                        
                        # Salva risultati
                        results_data.append({
                            'architecture': str(arch),
                            'activations': str(act),
                            'eta': eta,
                            'lambda': lam,
                            'alpha': alpha,
                            'accuracy': val_acc,
                            'epochs': epochs_run
                        })
                        
                    except Exception as e:
                        print(f"   -> ERRORE nella config {config_name}: {e}")

    # 4. SALVATAGGIO SU CSV
    df = pd.DataFrame(results_data)
    df = df.sort_values(by='accuracy', ascending=False)
    
    filename = 'grid_search_results_monk.csv'
    df.to_csv(filename, index=False)
    
    print(f"\n--- GRID SEARCH COMPLETATA ---")
    print(f"Migliori 3 configurazioni salvate in {filename}:")
    print(df.head(3))

if __name__ == "__main__":
    main()