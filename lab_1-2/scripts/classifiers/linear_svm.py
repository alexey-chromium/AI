from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation with loops.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels.
    - reg: regularization strength.

    Returns:
    - loss as single float
    - gradient with respect to weights W
    """
    dW = np.zeros(W.shape)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0

    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        for j in range(num_classes):
            if j == y[i]:
                continue

            margin = scores[j] - correct_class_score + 1

            if margin > 0:
                loss += margin

                # Gradient for incorrect class j
                dW[:, j] += X[i]

                # Gradient for correct class
                dW[:, y[i]] -= X[i]

    loss /= num_train
    dW /= num_train

    loss += reg * np.sum(W * W)
    dW += 2 * reg * W

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.
    """
    loss = 0.0
    dW = np.zeros(W.shape)

    num_train = X.shape[0]

    #########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    scores = X.dot(W)
    correct_class_scores = scores[np.arange(num_train), y].reshape(-1, 1)

    margins = scores - correct_class_scores + 1
    margins = np.maximum(0, margins)
    margins[np.arange(num_train), y] = 0

    loss = np.sum(margins) / num_train
    loss += reg * np.sum(W * W)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    binary = np.zeros(margins.shape)
    binary[margins > 0] = 1

    row_sum = np.sum(binary, axis=1)
    binary[np.arange(num_train), y] = -row_sum

    dW = X.T.dot(binary)
    dW /= num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW