# -*- coding: utf-8 -*-
"""
@author: Bannikov Maxim

"""

import numpy as np
import matplotlib.pyplot as plt
from google.colab import files

def get_data():
  w = np.array([-0.5, 0.5])
  b = 0.1
  X = np.random.random((300,2))*2 - 1
  Y = np.sign(X.dot(w) + b)
  return X, Y

class Perceptron:

  def fit(self, X, Y, lr = 1.0, epochs = 1000):
    D = X.shape[1]
    N = len(Y)
    self.w = np.random.randn(D)
    self.b = 0
    costs = []
    for i in range(epochs):
      P = self.predict(X)
      miss = np.nonzero(Y != P)[0]
      if len(miss) == 0:
        break
      sample = np.random.choice(miss)
      self.w += lr*Y[sample]*X[sample,:]
      self.b += lr*Y[sample]
      c = len(miss)/N
      costs.append(c)
    print('final w:', self.w, "final b:", self.b, "epochs:", (i + 1), "/", epochs)

  def predict(self, X):
    P = np.sign(X.dot(self.w) + self.b)
    return P

  def score(self, X, Y):
    P = self.predict(X)
    return np.mean(P == Y)

if __name__ == '__main__':
  X, Y = get_data()

  model = Perceptron()
  model.fit(X, Y)
  P = model.predict(X)
  print('Train accuracy:', model.score(X, Y))

  plt.scatter(X[:,0], X[:,1], c = Y)
  plt.savefig("perceptron.png")
  files.download("perceptron.png")

  plt.scatter(X[:,0], X[:,1], c = P)
  plt.savefig("jimmy_neutron.png")
  files.download("jimmy_neutron.png")