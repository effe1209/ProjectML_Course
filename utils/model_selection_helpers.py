import numpy as np

#helper function to count params that we will use later, since often we will get 100% acc, we chose the config with less parameters
def count_parameters(architecture):
    tot_params = 0
    for i in range(len(architecture) - 1):
        tot_params += architecture[i] * architecture[i+1] + architecture[i+1] # weights and bias
    return tot_params

# instability coefficient function
def instability_coeff(val_losses: list) -> float:
    """
    Takes val loss vec and every time val loss increases we add the diff, uses relative diff, otherwise mse would be biased
    """
    instability = 0.0
    for i in range(1, len(val_losses)):
        if val_losses[i] > val_losses[i - 1]:
            instability += (val_losses[i] - val_losses[i - 1]) / val_losses[i - 1]
    return instability

def tran_val_diff(train_losses: list, val_losses: list) -> float:
    """
    Computes the difference between train loss vec and val loss vec, only when loss train < val
    """
    return np.sum(np.maximum(0, np.array(val_losses) - np.array(train_losses)))