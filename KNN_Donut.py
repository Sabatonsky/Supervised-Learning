# -*- coding: utf-8 -*-
"""
@author: Bannikov Maxim

"""

import numpy as np
from google.colab import files
from sortedcontainers import SortedList
import matplotlib.pyplot as plt
from datetime import datetime

def get_donut():
  N = 200
  R_inner = 5
  R_outer = 10
  R1 = np.random.randn(int(N/2)) + R_inner
  theta = 2*np.pi*np.random.random(int(N/2))
  X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

  R2 = np.random.randn(int(N/2)) + R_outer
  theta = 2*np.pi*np.random.random(int(N/2))
  X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

  X = np.concatenate([X_inner, X_outer])
  Y = np.array([0]*(int(N/2)) + [1]*(int(N/2)))
  return X, Y

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
    Y = np.zeros(len(X))
    for i, x_test in enumerate(X):
      sl = SortedList()
      for j, x_train in enumerate(self.X):
        diff = x_test - x_train
        d = diff.dot(diff)
        if len(sl) < self.k:
          sl.add((d, self.Y[j]))
        else:
          if d < sl[-1][0]:
            del sl[-1]
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
  X, Y = get_donut()
  knn = KNN(3)
  t0 = datetime.now()
  knn.fit(X, Y)
  P = knn.predict(X)
  print('Train accuracy:', knn.score(X, Y))

plt.scatter(X[:,0], X[:,1], c = Y)
plt.savefig("donut.png")
files.download("donut.png")

plt.scatter(X[:,0], X[:,1], c = P)
plt.savefig("donut_1.png")
files.download("donut_1.png")