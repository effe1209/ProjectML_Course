def random_layer_init(input_dim: int, output_dim: int, std: float = 0.02) -> np.ndarray:
    """
    Creates the first weights of the neural network layer.

    Args:
        input_dim (int): Dimension of the input vector.
        output_dim (int): Dimension of the output vector.
        std (float, optional): Standard deviation of the normal distribution. Defaults to 0.02.

    Returns:
        np.ndarray: Matrix of the weights
    """
    return np.random.normal(loc=0.0, scale=std, size=(output_dim, input_dim))