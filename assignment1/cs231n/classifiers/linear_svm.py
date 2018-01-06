import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero
  # print("ini Shape:", dW.shape, W.shape)

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    # for each training example
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    count = 0
    for j in xrange(num_classes):
      # for each class
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        count += 1
        loss += margin
        dW[:, j] += X[i]
        # print("ffi Shape:", dW.shape)
    dW[:, y[i]] -=  count * X[i]
    # print("f Shape:", dW.shape)

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  # print("o Shape:", dW.shape)

  # Add regularization to the loss.
  # print("Loss before reg: ", loss)
  loss += 0.5 * reg * np.sum(W * W)
  # print("Loss after reg: ", loss)
  dW += reg * W
  # print("o2 Shape:", dW.shape)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################

  return loss, dW



def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

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
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################

  #############################################################################
  #                                 Dummy data                                #
  #############################################################################

  # d = 3
  # N = 6
  # c = 4
  #
  # X = np.array( range(N * d) )
  # X += 1
  # X = X.reshape(N, d)
  #
  # W = np.array( range(20, 20 + 2 * d * c, 2), dtype=float )
  # W /= max(W)
  # np.random.shuffle(W)
  # W = W.reshape(d, c)
  #
  # y = np.random.randint(0, high=c, size=N)

  #############################################################################
  #                                    IMPL                                   #
  #############################################################################

  num_train = X.shape[0]
  d = X.shape[1]
  c = W.shape[1]

  scores = X.dot(W)
  correct_class_scores = scores[range(y.shape[0]), y]
  all_losses = np.zeros(shape=(num_train, c))
  all_losses = 1 + scores - correct_class_scores[:, np.newaxis]
  all_losses = (all_losses > 0) * all_losses
  all_losses[range(y.shape[0]), y] = 0

  losses = np.sum(all_losses, axis=1)

  total_loss = sum(losses)
  total_loss /= num_train

  total_loss += 0.5 * reg * np.sum(np.square(W))

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  dW = np.zeros(shape=(d, c))

  # Partially vectorized code
  #
  # for i in xrange(num_train):
  #   count = np.sum(all_losses[i]>0)
  #   dW[:, all_losses[i]>0] += X[i][:, np.newaxis]
  #   dW[:, y[i]] -=  count * X[i]
  #
  # dW /= num_train
  # dW += reg * np.sum(W)

  # Completely vectorized code
  #
  # count = np.sum(all_losses > 0, axis=1)
  # print("Shapes: ", X.shape, count.shape, (X * count[:, np.newaxis]).shape)
  #
  # dW /= num_train
  # dW += reg * np.sum(W)

  # Fully vectorized version. Roughly 10x faster.
  # I gave up!
  # copied from https://github.com/bruceoutdoors/

  X_mask = np.zeros(all_losses.shape)
  X_mask[all_losses > 0] = 1
  incorrect_counts = np.sum(X_mask, axis=1)
  X_mask[np.arange(num_train), y] = -incorrect_counts
  dW = X.T.dot(X_mask)

  dW /= num_train  # average out weights
  dW += reg * W  # regularize the weights

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return total_loss, dW
