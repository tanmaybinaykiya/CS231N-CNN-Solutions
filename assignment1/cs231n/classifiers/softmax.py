import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax(x):
  f = x.copy()
  f -= np.max(x)
  return np.exp(f) / np.sum(np.exp(f))

def test_softmax():
  a = np.random.choice(1000, 56)
  print(a)
  a = softmax(a)
  print(a)
  print("Sum: ", sum(a))

# test_softmax()

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N, D = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  for i in xrange(N):
    xi = X[i]
    yi = y[i]

    scores = W.T.dot(xi)

    scores_map = softmax(scores)
    li = -np.log(scores_map[yi])
    loss += li
    for k in xrange(W.shape[1]):
      dW[:, k] += (scores_map[k] - (k == yi)) * xi

  loss /= N
  dW /= N

  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_mat(X):
  f = X.copy()
  f -= (np.max(X, axis= 1)[:, np.newaxis])
  expf = np.exp(f)
  return np.divide(expf, (np.sum(expf, axis=1)[:, np.newaxis]))

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_train = X.shape[0]

  scores_map = softmax_mat(X.dot(W))
  loss = np.mean(-np.log(scores_map[np.arange(num_train), y])) + 0.5 * reg * np.sum(np.square(W))

  mask = np.zeros_like(scores_map)
  mask[np.arange(X.shape[0]), y] = 1
  dW = X.T.dot(scores_map - mask) / num_train + reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def test_vec():
  d = 3
  N = 6
  c = 4

  X = np.array( range(N * d))
  X += 1
  X = X.reshape(N, d)

  W = np.array( range(20, 20 + 2 * d * c, 2), dtype=float )
  W /= max(W)
  np.random.shuffle(W)
  W = W.reshape(d, c)

  y = np.random.randint(0, high=c, size=N)

  loss_naive, _ = softmax_loss_naive(W, X, y, reg = 0.00005)
  loss_vec, _ = softmax_loss_vectorized(W, X, y, reg=0.00005)

  print("Naive: ", loss_naive, "loss_vec:", loss_vec)

