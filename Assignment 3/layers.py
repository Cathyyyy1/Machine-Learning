import numpy as np
from collections import OrderedDict


from activations import Activation



class FullyConnectedLayer():
    """
    A fully-connected layer multiplies its input by a weight matrix, adds
    a bias, and then applies an activation function.
    """

    def __init__(self, n_out, activation):
        self.activation = Activation(activation)
        
        self.n_in = None
        self.n_out = n_out

        self.parameters = {}
        self.cache = {}
        self.gradients = {}

    def _init_parameters(self, X_shape):
        """
        Initialize all layer parameters (weights, biases).
        """
        self.n_in = X_shape[1]
        W = np.random.normal(0, 1, size=(self.n_in, self.n_out))
        b = np.zeros((self.n_out,))[np.newaxis, :]

        self.parameters = OrderedDict({"W": W, "b": b})
        self.cache: OrderedDict = {}
        self.gradients: OrderedDict = OrderedDict({"W": np.zeros(W.shape), "b": np.zeros(b.shape)})

    def forward(self, X):
        """
        Forward pass: multiply by a weight matrix, add a bias, apply activation.
        Also, store all necessary intermediate results in the `cache` dictionary
        to be able to compute the backward pass.

        Parameters
        ----------
        X  input matrix of shape (batch_size, input_dim)

        Returns
        -------
        a matrix of shape (batch_size, output_dim) which is the output of the layer
        """

        # initialize layer parameters if they have not been initialized
        if self.n_in is None:
            self._init_parameters(X.shape)
            
        # perform an affine transformation and activation
        Z = X.dot(self.parameters["W"]) + self.parameters["b"]
        
        # Activation
        out = self.activation.forward(Z)
    
        # store information necessary for backprop in `self.cache`
        self.cache["X"] = X
        self.cache["Z"] = Z
        
        return out

    def backward(self, dLdY):
        """Backward pass for fully connected layer.
        Compute the gradients of the loss with respect to:
            1. the weights of this layer (mutate the `gradients` dictionary)
            2. the bias of this layer (mutate the `gradients` dictionary)
            3. the input of this layer (return this)

        Parameters
        ----------
        dLdY  gradient of the loss with respect to the output of this layer
              shape (batch_size, output_dim)

        Returns
        -------
        gradient of the loss with respect to the input of this layer
        shape (batch_size, input_dim)
        """
        ### BEGIN YOUR CODE ###
        
        # unpack the cache
        X = self.cache["X"]  # input, shape (batch_size, input_dim)
        Z = self.cache["Z"]  # pre-activation, shape (batch_size, output_dim)
        
        # get the weights
        W = self.parameters["W"]  # shape (input_dim, output_dim)
        
        # compute the gradients of the loss w.r.t. all parameters as well as the
        # input of the layer
        dLdZ = self.activation.backward(Z, dLdY)  # shape (batch_size, output_dim)
        
        # store the gradients in `self.gradients`
        # the gradient for self.parameters["W"] should be stored in
        # self.gradients["W"], etc.
        # compute gradients for weights: dL/dW = X^T * dL/dZ
        self.gradients["W"] = np.dot(X.T, dLdZ)  # shape (input_dim, output_dim)
        
        # compute gradients for bias: dL/db = sum(dL/dZ) over batch dimension
        self.gradients["b"] = np.sum(dLdZ, axis=0, keepdims=True)  # shape (1, output_dim)
        
        # compute gradients for input: dL/dX = dL/dZ * W^T
        dX = np.dot(dLdZ, W.T)  # shape (batch_size, input_dim)
        
        ### END YOUR CODE ###
        return dX

    
    def clear_gradients(self):
        self.cache = OrderedDict({a: [] for a, b in self.cache.items()})
        self.gradients = OrderedDict({a: np.zeros_like(b) for a, b in self.gradients.items()})

    def forward_with_param(self, param_name, X):
        """Call the `forward` method but with `param_name` as the variable with
        value `param_val`, and keep `X` fixed.
        """

        def inner_forward(param_val):
            self.parameters[param_name] = param_val
            return self.forward(X)

        return inner_forward

    def _get_parameters(self):
        return [b for a, b in self.parameters.items()]

    def _get_cache(self):
        return [b for a, b in self.cache.items()]

    def _get_gradients(self):
        return [b for a, b in self.gradients.items()]
    