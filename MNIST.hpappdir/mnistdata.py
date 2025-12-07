from array import *
from ustruct import *
from urandom import *
from hpprime import *

from utils import *

def readsamples(n, offset=0, f_images='train-images.idx3-ubyte', f_labels='train-labels.idx1-ubyte'):
  with open(f_images, 'rb') as f:
    f.seek(16 + offset * 784)
    X = []
    for i in range(n):
      b = bytearray(784)
      f.readinto(b)
      X.append(list(b))
      #output.append(array('B', bytes(f.read(rows * cols), 'utf-8')))
      #print(len(output[-1]))

  with open(f_labels, 'rb') as f:
    f.seek(8 + offset)
    y = []
    b = bytearray(n)
    f.readinto(b)
    for i in range(n):
      s = [0] * 10
      s[b[i]] = 1
      y.append(s)
  X = [[x / 255 for x in sample] for sample in X]
  return X, y

def get_preprocessed(n, offset=0):
  X, y = readsamples(n)
  s = getrandbits(32)
  seed(s)
  X = shuffle(X)
  seed(s)
  y = shuffle(y)
  index = int(.8 * n)
  X_trn = X[:index]
  y_trn = y[:index]
  X_vld = X[index:]
  y_vld = y[index:]
  return X_trn, y_trn, X_vld, y_vld

"""
X = readsamples(50)
for s in X:
  fillrect(0, 0, 0, 320, 240, 0, 0)
  for y in range(28):
    for x in range(28):
      c = 0x10101 * s[28 * y + x]
      fillrect(0, 8 * x, 8 * y, 8, 8, c, c)
  eval('wait')
"""