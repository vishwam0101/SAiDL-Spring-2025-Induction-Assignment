from builtins import object
import numpy as np

from helper_functions.layers import *
from helper_functions.fast_layers import *
from helper_functions.layer_utils import *



class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(
        self,
        input_dim=(3, 32, 32),
        num_filters=32,
        filter_size=7,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
        dtype=np.float32,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # Initialize weights and biases for the three-layer convolutional network. #
        #                                                                          #
        # Layer 1: Conv - ReLU - Pool                                              #
        ############################################################################
        C, H, W = input_dim
        F = num_filters
        HH, WW = filter_size, filter_size

        # Conv layer weights and biases
        self.params["W1"] = np.random.randn(F, C, HH, WW) * weight_scale
        self.params["b1"] = np.zeros(F)

        ############################################################################
        # Layer 2: Affine - ReLU                                                   #
        ############################################################################
        # Pooling reduces the size by 2 if pool size is 2x2 with stride 2
        H_out = H // 2  # After 2x2 pooling
        W_out = W // 2

        # Affine layer dimensions after flattening
        self.params["W2"] = np.random.randn(F * H_out * W_out, hidden_dim) * weight_scale
        self.params["b2"] = np.zeros(hidden_dim)

        ############################################################################
        # Layer 3: Affine - Softmax                                                #
        ############################################################################
        self.params["W3"] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params["b3"] = np.zeros(num_classes)

        ############################################################################
        # Store everything as dtype                                                #
        ############################################################################
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None, loss_type="ce"):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        - X: Input data of shape (N, C, H, W)
        - y: Vector of labels of shape (N,). If None, return class scores.
        - loss_type: Type of loss to compute ('ce', 'nce', 'apl')
        """
        W1, b1 = self.params["W1"], self.params["b1"]
        W2, b2 = self.params["W2"], self.params["b2"]
        W3, b3 = self.params["W3"], self.params["b3"]

        # Convolutional and pooling parameters
        filter_size = W1.shape[2]
        conv_param = {"stride": 1, "pad": (filter_size - 1) // 2}
        pool_param = {"pool_height": 2, "pool_width": 2, "stride": 2}

        # ===========================
        # ✅ Forward Pass
        # ===========================
        out1, cache1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out2, cache2 = affine_relu_forward(out1, W2, b2)
        scores, cache3 = affine_forward(out2, W3, b3)

        # ===========================
        # 🔍 Inference Mode: Return Scores
        # ===========================
        if y is None:
            return scores

        # ===========================
        # 🎯 Select Loss Type Dynamically
        # ===========================
        if loss_type == "ce":
            loss_value, dscores = softmax_loss(scores, y)
        elif loss_type == "nce":
            loss_value, dscores = nce_loss(scores, y)
        elif loss_type == "apl":
            # ✅ Correctly unpack 4 values from APL loss
            loss_value, dscores, active_loss_val, passive_loss_val = apl_loss(scores, y)
        else:
            raise ValueError(f"Invalid loss_type: {loss_type}")

        # ===========================
        # 📚 Add Regularization to Loss
        # ===========================
        loss_value += 0.5 * self.reg * (
            np.sum(W1**2) + np.sum(W2**2) + np.sum(W3**2)
        )

        # ===========================
        # 🔄 Backpropagation Using Correct dscores
        # ===========================
        dout2, dW3, db3 = affine_backward(dscores, cache3)
        dout1, dW2, db2 = affine_relu_backward(dout2, cache2)
        dX, dW1, db1 = conv_relu_pool_backward(dout1, cache1)

        # ===========================
        # 📦 Store Gradients
        # ===========================
        grads = {}
        grads["W1"], grads["b1"] = dW1 + self.reg * W1, db1
        grads["W2"], grads["b2"] = dW2 + self.reg * W2, db2
        grads["W3"], grads["b3"] = dW3 + self.reg * W3, db3

        # ✅ Return correct values for APL
        if loss_type == "apl":
            return loss_value, grads, active_loss_val, passive_loss_val
        else:
            return loss_value, grads

