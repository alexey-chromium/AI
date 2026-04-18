from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
from past.builtins import xrange


class TwoLayerNet(object):

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        hidden_linear = X.dot(W1) + b1
        hidden = np.maximum(0, hidden_linear)
        scores = hidden.dot(W2) + b2

        if y is None:
            return scores

        shifted_scores = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted_scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        correct_log_probs = -np.log(probs[np.arange(N), y])
        loss = np.sum(correct_log_probs) / N
        loss += reg * (np.sum(W1 * W1) + np.sum(W2 * W2))

        grads = {}

        dscores = probs.copy()
        dscores[np.arange(N), y] -= 1
        dscores /= N

        grads['W2'] = hidden.T.dot(dscores) + 2 * reg * W2
        grads['b2'] = np.sum(dscores, axis=0)

        dhidden = dscores.dot(W2.T)
        dhidden[hidden_linear <= 0] = 0

        grads['W1'] = X.T.dot(dhidden) + 2 * reg * W1
        grads['b1'] = np.sum(dhidden, axis=0)

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train // batch_size, 1)

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            batch_indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[batch_indices]
            y_batch = y[batch_indices]

            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']
            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            if it % iterations_per_epoch == 0:
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        scores = self.loss(X)
        y_pred = np.argmax(scores, axis=1)

        return y_pred