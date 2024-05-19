# -*- coding: utf-8 -*-
"""
@author: Bannikov Maxim

"""

import numpy as np
from google.colab import files
from sortedcontainers import SortedList
import matplotlib.pyplot as plt
from datetime import datetime

def get_xor():
  X = np.zeros((200, 2))
  X[:50] = np.random.random((50,2)) / 2 + 0.5
  X[50:100] = np.random.random((50,2)) / 2
  X[100:150] = np.random.random((50,2)) / 2 + np.array([0, 0.5])
  X[150:200] = np.random.random((50,2)) / 2 + np.array([0.5, 0])
  Y = np.array([0]*100 + [1]*100)
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
  X, Y = get_xor()
  knn = KNN(3)
  t0 = datetime.now()
  knn.fit(X, Y)
  P = knn.predict(X)
  print('Train accuracy:', knn.score(X, Y))

plt.scatter(X[:,0], X[:,1], c = Y)
plt.savefig("xor.png")
files.download("xor.png")

plt.scatter(X[:,0], X[:,1], c = P)
plt.savefig("xor_1.png")
files.download("xor_1.png")