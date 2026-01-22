import numpy as np
import copy
from model.network import NeuralNetwork
from model.losses import Loss
from utils import DataLoader
from model.activations import sigmoid
from model.losses import mee
from utils.plot_curves import plot_curves

class Trainer:
  """
  Trains the neural network
  """
  def __init__ (self,
                nn: NeuralNetwork,
                loss: Loss,
                X_train: np.ndarray,
                y_train: np.ndarray,
                X_val: np.ndarray = None,
                y_val: np.ndarray = None,
                epochs: int = 500,
                early_stopping: int = 100,
                min_improvement: float = 0.1,
                eta: float = 0.01,
                lam: float = 0.,
                alpha: float = 0.,
                batch_size: int = 8,
                shuffle_batches: bool = True,
                keep_last_batch:bool = True
                ):
    """
    Initializes the Trainer.

    Args:
    nn (NeuralNetwork): NeuralNetwork to be trained
    loss (Loss): Loss function to be used
    k_folds (int, optional): Number of k-folds for cross validation
    epochs (int, optional): Number of train epochs
    early_stopping (int, optional): Number of epochs to wait for early stopping
    eta (float, optional): Learning rate
    lam (float, optional): L2 regularization parameter
    alpha (float, optional): Momentum parameter
    """
    self.nn = nn
    self.loss = loss
    self.X_train = X_train
    self.y_train = y_train
    self.X_val = X_val
    self.y_val = y_val
    self.epochs = epochs
    self.early_stopping = early_stopping
    self.min_improvement = min_improvement
    self.eta = eta
    self.lam = lam
    self.alpha = alpha
    self.batch_size = batch_size
    self.shuffle_batches = shuffle_batches
    self.keep_last_batch = keep_last_batch

  def train(self, print_epochs: bool = False, plot_accuracy: bool = False, plot_mee : bool = False, plot_title: str= ''):
    """
    Trains the neural network based on the parameters passed to the constructor.
    """
    Val_exists = self.X_val is not None and self.y_val is not None

    # store the best neural network weights
    best_nn = copy.deepcopy(self.nn)
    best_loss = np.inf

    #early stop counter
    best_epoch_passed = 0

    # History
    train_loss_vec = []
    val_loss_vec = []

    train_acc_vec = []
    val_acc_vec = []

    train_mee_vec = []
    val_mee_vec = []
    # Initialize the datasets
    data_loader = DataLoader(X_dataset=self.X_train, y_dataset=self.y_train)

    # Train
    for epoch in range(self.epochs):
      batches = data_loader.get_batches(dataset = (self.X_train, self.y_train), batch_size = self.batch_size, shuffle = self.shuffle_batches, keep_last = self.keep_last_batch) #splits the dataset in batches, cant keep last because of L2
      train_loss = [] #keeps the count of train loss
      for x, y in batches:
        out = self.nn.forward(x)
        tr_loss = self.loss.compute_loss(out[-1][-1], y)
        train_loss.append(np.mean(tr_loss)) #adds to the count of train loss
        loss_grad = - self.loss.compute_loss_gradient(out[-1][-1], y) #has to be negative because we add gradients
        grad = self.nn.compute_gradients(out, loss_grad)
        b_size = x.shape[0]
        self.nn.gradient_descent(gradients = grad, b_size=b_size ,eta = self.eta, lam = self.lam, alpha = self.alpha)

      train_loss_vec.append(np.mean(train_loss)) #train loss mean
      #train acc
      if plot_accuracy:
        out = self.nn.forward(self.X_train)[-1][-1] 
        if self.loss.loss_f == 'binary cross entropy sigmoid':
          out = sigmoid(out)

        predictions = np.round(out)
        train_acc_vec.append(np.mean(predictions == self.y_train))
      #train mee
      if plot_mee:
        out = self.nn.forward(self.X_train)[-1][-1]  
        train_mee_vec.append(np.mean(mee(self.y_train, out)))
      
      #val loss
      if Val_exists:
        out = self.nn.forward(self.X_val)[-1][-1]
        val_loss =  self.loss.compute_loss(out, self.y_val) #computes the test loss
        val_loss_vec.append(np.mean(val_loss)) #val loss mean in the epoch
        #val acc
        if plot_accuracy:
          if self.loss.loss_f == 'binary cross entropy sigmoid':
            out = sigmoid(out)
          
          predictions = np.round(out)
          val_acc_vec.append(np.mean(predictions == self.y_val))

        #validation mee
        if plot_mee:
            out = self.nn.forward(self.X_val)[-1][-1]  
            val_mee_vec.append(np.mean(mee(self.y_val, out)))

        if np.mean(val_loss) < best_loss * (1 - self.min_improvement):
          best_loss = np.mean(val_loss) 
          best_nn = copy.deepcopy(self.nn)
          best_epoch_passed = 0
        else:
          best_epoch_passed += 1

      else: #a mali estremi, si usa la train loss per sapere il miglior modello, ma non deve essere preso "sul serio"
        if np.mean(train_loss) < best_loss * (1 - self.min_improvement):
          best_loss = np.mean(train_loss) 
          best_nn = copy.deepcopy(self.nn)
          best_epoch_passed = 0
        else:
          best_epoch_passed += 1

      #print epochs
      if print_epochs:
        print(f"epoch {epoch} ---------------------\ntrain loss: {np.mean(train_loss)}")
        if Val_exists:
          print(f"val loss: {np.mean(val_loss)}")

      # Break for early stopping
      if best_epoch_passed >= self.early_stopping:
        break
    if plot_accuracy and Val_exists:
      plot_curves(np.array(train_acc_vec), np.array(val_acc_vec), 'accuracy', 'test', title = plot_title, save_plots=True)
    
    if plot_mee and Val_exists:
      plot_curves(np.array(train_mee_vec), np.array(val_mee_vec), 'mee', 'test', title = plot_title, save_plots=True)

    return best_nn, train_loss_vec, val_loss_vec