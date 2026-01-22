import numpy as np
from model.layers import Layer

class NeuralNetwork:
  def __init__(self, layer_dimensions: list[int], layer_activations: list[str], std : float = None):
    """
    Initializes the NeuralNetwork.

    Args:
        layer_dimensions (list[int]): dimension of the layers in order.
        layer_activations (list[str]): activations of the layers in order.
        std (float, optional): standard deviation of the initialization of the weights
    """
    assert len(layer_dimensions) - 1 == len(layer_activations), "Dimensions -1 and activations must have the same length"
    self.layers = []

    # Create layers, initializa with Xavier or He if std is None
    for i in range(len(layer_dimensions)-1):
      dinamic_std = std
      
      if dinamic_std is None:
        dinamic_std = np.sqrt(2 / (layer_dimensions[i] + layer_dimensions[i+1])) #Xavier initialization
        if layer_activations[i] in ['relu', 'leaky relu']:
          dinamic_std = np.sqrt(2 / (layer_dimensions[i])) #He initialization for relu
      self.layers.append(Layer(layer_dimensions[i], layer_dimensions[i + 1], std = dinamic_std, act_f = layer_activations[i]))

    self.gradient_old = [np.zeros_like(layer.weights) for layer in self.layers] #previous gradients list, required by momentum.

  def forward(self, x: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Computes NeuralNetwork's layers' outputs.

    Args:
        x (np.ndarray): NeuralNetwork's input array (matrix of (batch_size, x_dim)).

    Returns:
        tuple[list[np.ndarray]]: List of nets and list of outs.
    """
    assert x.ndim == 2, "Input must be a matrix, if you want to pass a vector at a time, pass it with shape (1, x_shape)"
    out_list = [x] # (b_size, input_dim)
    net_list = []

    for layer in self.layers:
      net_list.append(layer.net(out_list[-1]))
      out_list.append(layer.output(net_list[-1]))

    return (net_list, out_list) #lists of "what happend to x", if you only want output of the nn do nn.forward[-1][-1]

  def compute_gradients(self, forward_x: tuple[list[np.ndarray], list[np.ndarray]], loss_grad: np.array) -> list[np.ndarray]:
    """
    Executes a full backpropagation cycle.

    Args:
        x (np.array): NeuralNetwork's input array.
        y (np.array): gradient of the loss.
        eta (float): learning rate parameter.
    """

    # Forward pass
    net_list = forward_x[0]
    out_list = forward_x[1]

    #Backward pass: compute the deltas for the output layer
    error_vector = loss_grad
    gradients = []
    for i in range(len(self.layers)):
      #error backprop
      grad_out = self.layers[-i - 1].out_derivative(net_list[- i -1])
      delta = error_vector * grad_out
      error_vector = self.layers[-i - 1].net_derivative(delta)

      #grad computing
      out = out_list[-i -2]
      ones_vec = np.ones((out.shape[0], 1))
      out = np.concatenate((ones_vec, out), axis = 1)
      gradient = delta.T @ out
      gradients.append(gradient)
    return gradients[::-1]#invert the list since we calculated them from the last to the first, lets return them in order

  def gradient_descent(self, gradients: list[np.ndarray], eta: float, b_size: int, lam: float = 0, alpha: float = 0) -> None:
    for i in range(len(self.layers)):
      w = np.array(self.layers[i].weights)
      w[:, 0] = 0.
      gradient_new = eta * (gradients[i] / b_size - 2 * lam * w) + alpha * self.gradient_old[i] #alpha, lambda and eta depend on each other. we also divide gradients magnitude by the batch size.
      self.layers[i].weights +=  gradient_new
      self.gradient_old[i] = gradient_new