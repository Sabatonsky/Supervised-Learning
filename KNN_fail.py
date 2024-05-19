# -*- coding: utf-8 -*-
"""
@author: Bannikov Maxim

"""

import numpy as np
from sortedcontainers import SortedList
from datetime import datetime

class KNN(object):
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
    self.Y = np.zeros(len(X))
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

  def score(self, X, Y):
    P = self.predict(X)
    return np.mean(P == Y)

def get_data():
  width = 8
  height = 8
  N = width * height
  X = np.zeros((N, 2))
  Y = np.zeros(N)
  n = 0
  start_t = 0
  for i in range(width):
    t = start_t
    for j in range(height):
      X[n] = [i, j]
      Y[n] = t
      n += 1
      t = (t + 1) / 2
    start_t = (start_t + 1) % 2
  return X, Y

if __name__ == '__main__':
  X, Y = get_data()

  for k in (1, 2, 3, 4, 5):
    knn = KNN(k)
    t0 = datetime.now()
    knn.fit(X, Y)
    print("\n")
    print("Calculation for", k, "nearest nodes")
    print('Train time:', (datetime.now() - t0))
    t0 = datetime.now()
    print('Train accuracy:', knn.score(X, Y))
    print('Time to compute train accuracy:', datetime.now() - t0, "Train size:", len(Y))