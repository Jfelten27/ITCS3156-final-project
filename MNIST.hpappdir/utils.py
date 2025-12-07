from math import *
from urandom import *
from linalg import *

def shuffle(l: list):
  lc = l.copy()
  output = []
  while lc:
    output.append(lc.pop(randint(0, len(lc) - 1)))
  return output

def apply(f, m):
  if isinstance(m, list):
    output = []
    for x in m:
      #output.append([])
      output.append(apply(f, x))
    return output
  else:
    return f(m)
    try:
      return f(m)
    except:
      print(m)

def hmul(m1, m2):
  if shape(m1) != shape(m2):
    raise ValueError(str(shape(m1)) + ' * ' + str(shape(m2)))
  return [[x1 * x2 for x1, x2 in zip(*rows)] for rows in zip(m1, m2)]

def flatten(x):
  return sum(x, [])

def plot(x, y):
  #fillrect(0, 0, 0, 320, 240, 0, 0)
  #dimgrob(1, 320, 240, 0)
  for ix, iy in zip(x, y):
    pixon_c(1, ix, iy, 255)

def mean_std(X):
  n, m = shape(X)
  xf = flatten(X)
  xl = [xf[i::m] for i in range(m)]
  μ = [sum(x) / len(x) for x in xl]
  σ = [sqrt(sum([(_x - _mu)**2 for _x in x]) / n) for x, _mu in zip(xl, μ)]
  return μ, σ

def standardize(X, μ, σ):
  return hmul(sub(X, [μ] * len(X)), [[1 / s for s in σ]] * len(X))

def destandardize(X, μ, σ):
  return add(hmul(X, [σ] * len(X)), [μ] * len(X))

def nll(y, probs):
  return -1/len(y) * sum([sum(x) for x in hmul(y, apply(log, probs))])

def argmax(z):
  return z.index(max(z))