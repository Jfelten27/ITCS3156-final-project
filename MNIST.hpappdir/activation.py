from math import *
from linalg import *
from utils import *

class ReLU:
  @staticmethod
  def f(z):
    #return z
    return max(z, .000 * z)

  @staticmethod
  def df(z):
    #return ones(*shape(z))
    return int(z > 0) - .000 * int(z < 0)

class Linear:
  @staticmethod
  def f(z):
    return z

  @staticmethod
  def df(z):
    return ones(*shape(z))

class Tanh:
  @staticmethod
  def f(z):
    return tanh(z)

  @staticmethod
  def df(z):
    return 1 - tanh(z)**2

class Sigmoid:
  @staticmethod
  def f(z):
    return 1 / (1 + exp(-z))

  @staticmethod
  def df(z):
    return Sigmoid.f(z) * (1 - Sigmoid.f(z))

class Softmax:
  @staticmethod
  def f(z):
    if shape(z)[1] is None:
      z = [z]
    zmax = [[max(row)] * len(row) for row in z]
    #print(shape(z), shape(zmax))
    zexp = apply(exp, sub(z, zmax))
    return [[_z / sum(row)+.00001 for _z in row] for row in zexp]

  @staticmethod
  def df(z):
    raise NotImplementedError