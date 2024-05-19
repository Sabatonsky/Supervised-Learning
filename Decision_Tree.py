# -*- coding: utf-8 -*-
"""
@author: Bannikov Maxim

"""

import numpy as np
from sklearn.utils import shuffle
from sklearn.datasets import load_digits
import pandas as pd
from datetime import datetime

def entropy(y):
  N = len(y)
  s1 = (y == 1).sum() # Взяли все значения массива у и посчитали сумму всех в 1 категории
  if 0 == s1 or N == s1: # Проверяем, не принадлежат ли у одному классу.
    return 0
  p1 = float(s1) / N # Допустим есть 2 разных класса. считаем вероятность каждого в рамках нашей train выборки.
  p0 = 1 - p1
  return - p0*np.log2(p0) - p1*np.log2(p1) # рассчитываем общую энтропию

class TreeNode:
  def __init__(self, depth=0, max_depth=None):
    self.depth = depth # сколько сплитов данных мы уже сделали
    self.max_depth = max_depth # Максимальное количество сплитов (необходимо, чтобы не переобучить модель. С учетом того, что количество х в массиве у нас велико, decision tree в тупую нащупает тот самый)
    # семпл из N и просто предскажет его. Нам нужно, не чтобы он выискал какой-то крайний случай, который может быть исключением и выдал нам его, а чтобы машина давала решение на основании более крупных паттернов.
  def fit(self, X, Y):
    if len(Y) == 1 or len(set(Y)) == 1: #Если у нас всего 1 семпл, то выбор очевиден.
      self.col = None
      self.split = None
      self.left = None
      self.right = None
      self.prediction = Y[0] # Называем этот семпл
    else:
      D = X.shape[1] # Количество исходных X
      max_ig = 0 # Ищем какой сплит даст наибольший information gain
      for col in range(D):
        ig, split = self.find_split(X, Y, col) #Ищем на основании какого х будем пилить таблицу
        if ig > max_ig: # Нашли inform gain превышающий текущий максимум = сохраняем его как новый максимум
          max_ig = ig
          best_col = col
          best_split = split

      if max_ig == 0: # Если information gain по остаточной таблице меньше или равен 0 (Узнать больше ничего не получится, копать дальше не имеет смысла)
        self.col = None
        self.split = None
        self.left = None
        self.right = None
        self.prediction = np.round(Y.mean()) # Предсказываем среднее по оставшейся больнице с округлением.
      else:
        self.col = best_col # сохраняем данные для ветвления.
        self.split = best_split # х у нас не дескретный, поэтому нам нужно сохранять конкретное место сплита.

        if self.depth == self.max_depth: #Проверяем, дошли ли мы до максимальной глубины (защита от переобучения)
          self.left = None
          self.right = None
          # делаем предсказание для последнего сплита. Предсказания 2, в зависимости от того, какой X у нас выпадет.
          self.prediction = [
              np.round(Y[X[:,best_col] < self.split].mean()),
              np.round(Y[X[:,best_col] >= self.split].mean())
          ]
        else:
          left_idx = (X[:, best_col] < best_split) #Вызываем ноды сплита, строим дерево опций. По одной ветке для каждой опции.
          Xleft = X[left_idx]
          Yleft = Y[left_idx]
          self.left = TreeNode(self.depth + 1, self.max_depth)
          print('created node, current depth:', self.depth + 1, "/" ,self.max_depth)
          self.left.fit(Xleft, Yleft)

          right_idx = (X[:, best_col] >= best_split)
          Xright = X[right_idx]
          Yright = Y[right_idx]
          self.right = TreeNode(self.depth + 1, self.max_depth)
          self.right.fit(Xright, Yright)

  def find_split(self, X, Y, col):
    x_values = X[:,col] # берем все х из колонки, которую мы тестируем
    sort_idx = np.argsort(x_values) # делаем массив индексов, отсортированных по возрастанию в соответствии с колонкой, которую мы тестируем
    x_values = x_values[sort_idx] # сортируем х
    y_values = Y[sort_idx] # сортируем у

    boundaries = np.nonzero(y_values[:-1] != y_values[1:])[0] # индексы всех значений у, где перещелкивается категория.
    best_split = None
    max_ig = 0
    for i in boundaries: # Для каждого перещелкивания категорий
      split = (x_values[i] + x_values[i+1]) / 2 #Ищем сплит, который поделит категории в данной точке
      ig = self.information_gain(x_values, y_values, split) #Вызываем функцию, которая найдет information gain для конкретной точки
      if ig > max_ig: # Если это самая лучшая точка сплита, запоминаем её
        max_ig = ig
        best_split = split
    return max_ig, best_split # Возвращаем лучшую точку для конкретного X

  def information_gain(self, x, y, split): # Функция поиска information gain
    y0 = y[x < split] # Делим классификацию у на два промежутка: до сплита и после сплита.
    y1 = y[x >= split]
    N = len(y)
    y0len = len(y0) # считаем сколько у нас значений слева и справа от сплита
    if y0len == 0 or y0len == N: # Крайний случай, одна из сторон сплита нулевая.
      return 0
    p0 = float(len(y0)) / N
    p1 = 1 - p0
    return entropy(y) - p0*entropy(y0) - p1*entropy(y1) #Считаем энтропию по формуле энтропии. Вызываем внешнюю функцию entropy, для энтропии прошлого уровня и текущей ноды.

  def predict_one(self, x): # Мы выстроили рекурентную структуру нод. теперь мы идем по этой сети в поисках результата для тестового сета.
    if self.col is not None and self.split is not None: # У нас имеется значение лучшего столбца и лучшего сплита этого столбца для каждой конкретной ноды. Вынимаем для текущей.
      feature = x[self.col] # Ищем нужный столбец в тестовом сете, предсказания идут по одной строке за раз.
      if feature < self.split: # Если параметр столбца ниже значения для сплита идем налево
        if self.left: # проверяем наличие следующей ноды слева
          p = self.left.predict_one(x) # Если она есть, идем туда за предсказанием
        else:
          p = self.prediction[0] # Если ноды слева нет, то предсказываем самый вероятный сценарий

      if feature >= self.split: #Справа аналогично
        if self.right:
          p = self.right.predict_one(x)
        else:
          p = self.prediction[1]
    else:
      p = self.prediction
    return p # Предсказание возвращаем.

  def predict(self, x): # Обертка для predict one, перебирает массив test по строкам и составляет вектор предсказаний
    N = len(x)
    p = np.zeros(N)
    for i in range(N):
      p[i] = self.predict_one(x[i])
    return p

class DecisionTree: # Обертка обертки
  def __init__(self, max_depth = None):
    self.max_depth = max_depth

  def fit(self, X, Y): #Создает первую ноду и запускает фит функцию
    self.root = TreeNode(max_depth = self.max_depth)
    self.root.fit(X,Y)

  def predict(self, X): #запускает предсказание
    return self.root.predict(X)

  def score(self, X, Y): # измеряет accuracy
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

  idx_train = np.logical_or(Y_train == 1, Y_train == 0)
  idx_test = np.logical_or(Y_test == 1, Y_test == 0)

  X_train = X_train[idx_train]
  Y_train = Y_train[idx_train]
  X_test = X_test[idx_test]
  Y_test = Y_test[idx_test]

  model = DecisionTree(5)
  print(model.max_depth)
  t0 = datetime.now()
  model.fit(X_train, Y_train)
  p = model.predict(X_test)

  print('Train accuracy:', model.score(X_train, Y_train))

  print('Test accuracy:', model.score(X_test, Y_test))