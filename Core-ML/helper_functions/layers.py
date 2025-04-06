from builtins import range
import numpy as np


import numpy as np

def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    # Get input dimensions
    N = x.shape[0]  # Number of samples
    D = np.prod(x.shape[1:])  # Flatten the rest: d_1 * d_2 * ... * d_k
    
    # Reshape x to (N, D) to flatten the input
    x_reshaped = x.reshape(N, D)
    
    # Perform affine transformation: out = xW + b
    out = x_reshaped.dot(w) + b
    
    # Store values for backward pass
    cache = (x, w, b)
    
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    N = x.shape[0]

    # Reshape x to shape (N, D) where D = d_1 * ... * d_k
    x_reshaped = x.reshape(N, -1)

    # Gradient of loss w.r.t. bias (sum across all data points)
    db = np.sum(dout, axis=0)

    # Gradient of loss w.r.t. weights
    dw = x_reshaped.T.dot(dout)

    # Gradient of loss w.r.t. input x, reshape it back to original dimensions
    dx = dout.dot(w.T).reshape(x.shape)

    return dx, dw, db



def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    ###########################################################################
    # Implementing ReLU: max(0, x) element-wise                               #
    ###########################################################################
    out = np.maximum(0, x)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache



def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    x = cache
    ###########################################################################
    # Implementing gradient of ReLU: only pass gradients where x > 0         #
    ###########################################################################
    dx = dout * (x > 0)

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx



def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C)
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i]

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx



def nce_loss(x, y, epsilon=1e-5):
    """
    Compute the normalized cross-entropy (NCE) loss and its gradient.

    Inputs:
    - x: Input scores, shape (N, C), where x[i, j] is the score for the j-th class for the i-th input.
    - y: Vector of labels, shape (N,), where y[i] is the correct label for x[i].
    - epsilon: Small constant for numerical stability.

    Returns:
    - loss: Scalar giving the loss.
    - dx: Gradient of the loss with respect to x.
    """
    # ✅ Compute Softmax Probabilities
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)

    # ✅ Compute NCE Loss Correctly
    N = x.shape[0]
    true_class_probs = np.clip(probs[np.arange(N), y], epsilon, 1.0)
    loss = -np.sum(np.log(true_class_probs)) / N

    # ✅ Gradient of the Loss (Softmax Derivative)
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N

    return loss, dx






def mae_loss(x, y):
    """
    Compute Mean Absolute Error (MAE) loss and gradient.
    """
    N, C = x.shape
    true_scores = x[np.arange(N), y].reshape(-1, 1)

    # MAE Loss
    loss = np.sum(np.abs(x - true_scores)) / N

    # Gradient with respect to x
    dx = np.sign(x - true_scores) / N
    dx[np.arange(N), y] -= np.sum(np.sign(x - true_scores), axis=1) / N

    return loss, dx



def apl_loss(scores, y, alpha=0.5):
    """
    Compute Active-Passive Loss (APL) with CE/NCE as active loss and MAE as passive loss.
    """
    # ✅ Active Loss (CE/NCE)
    active_loss_val, dactive = softmax_loss(scores, y)

    # ✅ Passive Loss (MAE)
    passive_loss_val, dpassive = mae_loss(scores, y)  # Use corrected MAE gradient

    # ✅ Combined APL Loss
    loss = alpha * active_loss_val + (1 - alpha) * passive_loss_val
    dscores = alpha * dactive + (1 - alpha) * dpassive

    return loss, dscores, active_loss_val, passive_loss_val










def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    Inputs:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift parameter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'
      - eps: Constant for numeric stability
      - momentum: Momentum for running averages
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var: Array of shape (D,) giving running variance of features

    Returns:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == "train":
        # Compute mean and variance
        sample_mean = np.mean(x, axis=0)
        sample_var = np.var(x, axis=0)

        # Normalize input
        x_hat = (x - sample_mean) / np.sqrt(sample_var + eps)

        # Scale and shift
        out = gamma * x_hat + beta

        # Update running averages
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        # Cache values for backprop
        cache = (x, x_hat, sample_mean, sample_var, gamma, beta, eps)

    elif mode == "test":
        # Normalize using running averages
        x_hat = (x - running_mean) / np.sqrt(running_var + eps)

        # Scale and shift
        out = gamma * x_hat + beta

        # No cache for test mode
        cache = (None,)

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store updated running averages in bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache



def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    x, x_hat, sample_mean, sample_var, gamma, beta, eps = cache
    N, D = dout.shape

    # Gradients of beta and gamma
    dbeta = np.sum(dout, axis=0)
    dgamma = np.sum(dout * x_hat, axis=0)

    # Gradient through scale and shift
    dx_hat = dout * gamma

    # Backprop through normalization
    std_inv = 1.0 / np.sqrt(sample_var + eps)
    dvar = np.sum(dx_hat * (x - sample_mean) * -0.5 * std_inv**3, axis=0)
    dmean = np.sum(dx_hat * -std_inv, axis=0) + dvar * np.mean(-2.0 * (x - sample_mean), axis=0)

    # Backprop through mean and variance to input
    dx = dx_hat * std_inv + dvar * 2.0 * (x - sample_mean) / N + dmean / N

    return dx, dgamma, dbeta



def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    #############################b##############################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer."""
    
    # Extract shapes and dimensions
    N, C, H, W = x.shape         # Input dimensions
    F, _, HH, WW = w.shape       # Filter dimensions
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    # Calculate output dimensions
    H_out = 1 + (H + 2 * pad - HH) // stride
    W_out = 1 + (W + 2 * pad - WW) // stride
    
    # Pad the input along height and width with zeros
    x_padded = np.pad(
        x, 
        ((0, 0), (0, 0), (pad, pad), (pad, pad)),  # Pad only height and width
        mode='constant', 
        constant_values=0
    )
    
    # Initialize output with zeros
    out = np.zeros((N, F, H_out, W_out))
    
    # Perform convolution
    for n in range(N):  # Loop over all samples
        for f in range(F):  # Loop over all filters
            for i in range(H_out):
                for j in range(W_out):
                    # Define the region in the padded input
                    h_start = i * stride
                    h_end = h_start + HH
                    w_start = j * stride
                    w_end = w_start + WW
                    
                    # Extract the region and convolve
                    x_slice = x_padded[n, :, h_start:h_end, w_start:w_end]
                    out[n, f, i, j] = np.sum(x_slice * w[f]) + b[f]
    
    # Store values needed for backward pass
    cache = (x, w, b, conv_param)
    return out, cache



def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives of shape (N, F, H', W')
    - cache: A tuple of (x, w, b, conv_param) from conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, C, H, W)
    - dw: Gradient with respect to w, of shape (F, C, HH, WW)
    - db: Gradient with respect to b, of shape (F,)
    """
    # Unpack the cache
    x, w, b, conv_param = cache
    
    # Get dimensions
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    _, _, H_out, W_out = dout.shape
    
    # Get stride and padding values
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    # Pad the input and initialize gradient of x (padded)
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    dx_pad = np.zeros_like(x_pad)
    
    # Initialize gradients for w and b
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    
    # Loop over every image
    for n in range(N):  # Loop over each input image
        for f in range(F):  # Loop over each filter
            for i in range(H_out):  # Loop over output height
                for j in range(W_out):  # Loop over output width
                    
                    # Define boundaries of receptive field
                    vert_start = i * stride
                    vert_end = vert_start + HH
                    horiz_start = j * stride
                    horiz_end = horiz_start + WW
                    
                    # Slice the padded input
                    x_slice = x_pad[n, :, vert_start:vert_end, horiz_start:horiz_end]
                    
                    # Update gradients
                    dx_pad[n, :, vert_start:vert_end, horiz_start:horiz_end] += w[f] * dout[n, f, i, j]
                    dw[f] += x_slice * dout[n, f, i, j]
                    db[f] += dout[n, f, i, j]
                    
    # Remove padding from dx to match original input dimensions
    dx = dx_pad[:, :, pad:-pad, pad:-pad] if pad > 0 else dx_pad
    
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """

    # Get input dimensions
    N, C, H, W = x.shape

    # Get pooling parameters
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    # Compute output dimensions
    H_out = 1 + (H - pool_height) // stride
    W_out = 1 + (W - pool_width) // stride

    # Initialize output with zeros
    out = np.zeros((N, C, H_out, W_out))

    # Loop over all dimensions
    for n in range(N):  # Loop over each image in the batch
        for c in range(C):  # Loop over each channel
            for i in range(H_out):  # Loop over height
                for j in range(W_out):  # Loop over width
                    # Define region boundaries
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width

                    # Extract the region of interest
                    x_slice = x[n, c, h_start:h_end, w_start:w_end]

                    # Perform max-pooling (take max from the region)
                    out[n, c, i, j] = np.max(x_slice)

    # Store the input and pooling parameters for backpropagation
    cache = (x, pool_param)

    return out, cache



def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """

    # Unpack cache
    x, pool_param = cache

    # Get dimensions
    N, C, H, W = x.shape
    _, _, H_out, W_out = dout.shape

    # Get pooling parameters
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']

    # Initialize gradient w.r.t x to zero
    dx = np.zeros_like(x)

    # Loop through dimensions
    for n in range(N):  # Loop over each image in the batch
        for c in range(C):  # Loop over each channel
            for i in range(H_out):  # Loop over output height
                for j in range(W_out):  # Loop over output width
                    # Define the region in x that was pooled
                    h_start = i * stride
                    h_end = h_start + pool_height
                    w_start = j * stride
                    w_end = w_start + pool_width

                    # Get the slice from x
                    x_slice = x[n, c, h_start:h_end, w_start:w_end]

                    # Create a mask identifying the max location
                    mask = (x_slice == np.max(x_slice))

                    # Distribute the upstream gradient to the max location
                    dx[n, c, h_start:h_end, w_start:w_end] += dout[n, c, i, j] * mask

    return dx



def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with batch norm settings

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    N, C, H, W = x.shape

    # Reshape and transpose to apply batch norm
    x_transposed = x.transpose(0, 2, 3, 1)  # (N, H, W, C)
    x_flat = x_transposed.reshape(-1, C)    # (N * H * W, C)

    # Apply batch normalization to flattened input
    out_flat, cache = batchnorm_forward(x_flat, gamma, beta, bn_param)

    # Reshape back to original shape
    out_transposed = out_flat.reshape(N, H, W, C)
    out = out_transposed.transpose(0, 3, 1, 2)  # (N, C, H, W)

    return out, cache



def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    N, C, H, W = dout.shape

    # Reshape dout to (N*H*W, C) to apply vanilla batchnorm
    dout_reshaped = dout.transpose(0, 2, 3, 1).reshape(-1, C)

    # Backward pass using vanilla batchnorm
    dx_reshaped, dgamma, dbeta = batchnorm_backward(dout_reshaped, cache)

    # Reshape dx back to (N, C, H, W)
    dx = dx_reshaped.reshape(N, H, W, C).transpose(0, 3, 1, 2)

    return dx, dgamma, dbeta



def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
