# -*- coding: utf-8 -*-
"""
@author: Bannikov Maxim

"""

import numpy as np
from sklearn.utils import shuffle
from sklearn.datasets import load_digits
from scipy.stats import multivariate_normal
import pandas as pd
from datetime import datetime

class nbayes(object):
  def __init__(self):
    self.l = 10e-10
    self.dict_of_gaussians = {}
    self.priors = {}

  def fit(self, X, Y):
    self.dict_of_gaussians = {}
    self.priors = {}
    self.cat = set(Y)
    for c in self.cat:
      Xc = X[Y == c]
      mean, var = np.mean(Xc, axis = 0), np.var(Xc, axis = 0) + self.l
      self.dict_of_gaussians[c] = [mean, var]
      self.priors[c] = Xc.shape[0] / X.shape[0]

  def predict(self, X):
    N, D = X.shape
    K = len(self.dict_of_gaussians)
    P = np.zeros((N, K))
    for c, g in self.dict_of_gaussians.items():
      mean, var = self.dict_of_gaussians[c]
      P[:, c] = multivariate_normal.logpdf(X, mean=mean, cov=var) + np.log(self.priors[c])
    return np.argmax(P, axis = 1)

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

  bayes = nbayes()
  t0 = datetime.now()
  bayes.fit(X_train, Y_train)

  print('Train accuracy:', bayes.score(X_train, Y_train))

  print('Test accuracy:', bayes.score(X_test, Y_test))