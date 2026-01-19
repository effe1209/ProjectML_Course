import numpy as np

from model.activations import identity, identity_derivative, relu, relu_derivative, leaky_relu, leaky_relu_derivative, sigmoid, sigmoid_derivative, tanh, tanh_derivative
from model.layers.random_layer import random_layer_init

class Layer:
    """
    A fully connected layer, computes forward pass with weights and bias and activation function, then computes backward.
    """
    act_dictionary = {'identity': identity, 'relu': relu, 'leaky relu': leaky_relu,'sigmoid': sigmoid, 'tanh': tanh}
    act_derivative_dictionary = {'identity': identity_derivative, 'relu': relu_derivative, 'leaky relu': leaky_relu_derivative,'sigmoid': sigmoid_derivative, 'tanh': tanh_derivative}

    def __init__(self, input_dim: int, output_dim: int, initalization_method : str = 'rand_gauss', std: float = 0.02, act_f: str = 'relu'):
        """
        Initializes the Layer

        Args:
            input_dim (int): Dimension of the input vector.
            output_dim (int): Dimension of the output vector.
            initalization_method(str, optional): how to initialize the weights, default is 'rand_gauss'
            std (float, optional): parameter of random_layer_init
            act_f (str, optional): Activation function. Defaults to 'relu'.

        """
        if initalization_method == 'rand_gauss':
          self.weights = random_layer_init(input_dim + 1, output_dim, std = std) #weights, input_dim + 1 for the bias term

        assert act_f in ['identity', 'relu', 'leaky relu','sigmoid','tanh'], "Activation function not supported"
        self.act_f = act_f

    def output(self, net: np.ndarray) -> np.ndarray:
        """
        Plugs the net into the activation function and returns the "final" output of the layer.

        Args:
            net (np.ndarray): The net input array to the layer.

        Returns:
            np.ndarray: Net passed through the act. function.
        """
        return self.act_dictionary[self.act_f](net)

    def out_derivative(self, net: np.ndarray) -> np.ndarray:
        """
        Plugs the net into the activation function derivatrive and returns the derivative of output of the layer.

        Args:
            net (np.ndarray): The net input array to the layer.

        Returns:
            np.ndarray: Net passed through the derviv act. function.
        """
        return self.act_derivative_dictionary[self.act_f](net)


    def net(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Computes the net for the next layer.

        Args:
            input_vector (np.ndarray): The vector of the state., shape (batch_size, input_dim)

        Returns:
            np.ndarray: computed net array.
        """
        bias_ones =  np.ones((input_vector.shape[0], 1)) #vettore (batch_size, 1)
        input = np.concatenate((bias_ones, input_vector), axis = 1) #adding a "1" column for the bias term to the input.(batch_size, n_input)->(batch_size, n_input + 1)
        return (self.weights @ input.T).T #output(batch_size, output_dim)

    def net_derivative(self, delta: np.ndarray) -> np.ndarray:
        """
        Backpropagates the error layer (-> .pdf di Micheli "Backprop. notes latex")

        Args:
            delta (np.ndarray): Delta vector of the layer.

        Returns:
            np.ndarray: Backprop error.
        """
        w = self.weights[:, 1:] #remove the first colum, it was used for bias computation
        return delta @ w