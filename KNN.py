# -*- coding: utf-8 -*-
"""
@author: Bannikov Maxim

"""

import numpy as np
from sortedcontainers import SortedList
from sklearn.utils import shuffle
from sklearn.datasets import load_digits
import pandas as pd
from datetime import datetime

class knn(object):
  # array size for sorted array. Based on number of train samples.
  def __init__(self, k):
    self.k = k
  # add all data to memory
  def fit(self, X, Y):
    self.X = X
    self.Y = Y
  # calculate prediction by distance
  def predict(self, X):
    # set predictions to zero
    Y = np.zeros(len(X))
    # loop through list of i and x_predict tuples, [(0, x1), (1, x2) and so on]
    for i, x in enumerate(X):
    # set sorted list to fill it with distances between certain! x_predict and all! x_train
      sl = SortedList()
      # loop through list of j and x_train tuples, [(0, x1), (1, x2) and so on]
      for j, xt in enumerate(self.X):
        # grab 1 x_train multidim dot and 1 x_predict multidim dot. Both dots has equal num of dimentions, therefore we can find difference between this dots.
        diff = x - xt
        # find square distance aka MSE?
        d = diff.dot(diff)
        # compare current sl length with number of trained samples
        if len(sl) < self.k:
          # if we did not get enough samples, we are filling list
          sl.add((d, self.Y[j]))
        else:
          # Now we have collected k dots, and now our task is to replace farest dot by another, that is closer to x_predict
          if d < sl[-1][0]:
            del sl[-1]
            # assign classification type to our new dot
            sl.add((d, self.Y[j]))

      votes = {}
      for _, v in sl:
        votes[v] = votes.get(v, 0) + 1
      max_votes = 0
      max_votes_class = -1
      for v, count in votes.items():
        if count > max_votes:
          max_votes = count
          max_votes_class = v
      Y[i] = max_votes_class
    return Y

  def score(self, X, Y):
    P = self.predict(X)
    return np.mean(P == Y)

if __name__ == '__main__':
  mnist = load_digits()
  X = pd.DataFrame(mnist.data).to_numpy(dtype='float32')
  Y = pd.DataFrame(mnist.target).to_numpy().flatten()
  X_train = X[:1500,:] / 255
  X_test = X[1500:,:] / 255
  Y_train = Y[:1500]
  Y_test = Y[1500:]

  X_train, Y_train = shuffle(X_train, Y_train)
  X_test, Y_test = shuffle(X_test, Y_test)
  for k in (1, 2, 3, 4, 5):
    knn = knn(k)
    t0 = datetime.now()
    knn.fit(X_train, Y_train)
    print("\n")
    print("Calculation for", k, "nearest nodes")
    print('Train time:', (datetime.now() - t0))
    t0 = datetime.now()
    print('Train accuracy:', knn.score(X_train, Y_train))
    print('Time to compute train accuracy:', datetime.now() - t0, "Train size:", len(Y_train))
    t0 = datetime.now()
    print('Test accuracy:', knn.score(X_test, Y_test))
    print('Time to compute test accuracy:', datetime.now() - t0, "Test size:", len(Y_test))

