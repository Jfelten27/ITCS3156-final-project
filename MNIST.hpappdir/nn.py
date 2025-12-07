from math import *
from urandom import *
from hpprime import *
import linalg
from linalg import *
from matplotl import *

from utils import *
from activation import *

class Module:
  def __init__(self):
    self.w = None
    self.b = None

  def forward(self, *args):
    pass

class Perceptron(Module):
  def __init__(self, nIn, nOut, activ):
    self.w = apply(lambda x: (x - .5) * sqrt(2 / nIn), rand(nIn, nOut))
    self.b = apply(lambda x: (x - .5), rand(1, nOut))
    self.activ = activ
    #self.f = activ.f
    #self.df = activ.df

  def forward(self, X):
    #print(shape(dot(X, self.w)), shape(self.b))
    z = add(dot(X, self.w), [row.copy() for row in self.b * shape(X)[0]])
    #print(type(self.activ.f))
    if self.activ is Softmax:
      return z, self.activ.f(z)
    else:
      return z, apply(self.activ.f, z)

  def backward(self, X, z, da):
    if self.activ in (Linear, Softmax):
      dz = da
    else:
      dz = hmul(da, apply(self.activ.df, z))
    #dz = da
    #print('dz', shape(dz), 'w', shape(self.w))
    dw = mul(dot(transpose(X), dz), 1 / len(z))
    db = mul(dot(ones(1, shape(dz)[0]), dz), 1 / len(z))
    da = dot(dz, transpose(self.w))
    #print(dz[0], self.w[0])
    return dw, db, da #mul(m, transpose(self.w))

class NeuralNet(Module):
  def __init__(self, alpha=.02, epochs=500, batchsize=32, sizes=[1, 2, 1]):
    self.alpha = alpha
    self.epochs = epochs
    self.batchsize = batchsize
    #sizes = [1] + [8, 10] * 4 + [1]
    #sizes = [1, 2, 1]
    #sizes = [3, 32, 16, 1]
    self.layers = []
    if sizes == None:
        return
    for i in range(len(sizes) - 1):
      if i == len(sizes) - 2:
        activ = Softmax
      else:
        activ = ReLU
      self.layers.append(Perceptron(sizes[i], sizes[i + 1], activ))

  def forward(self, X):
    a = X
    lx = []
    lz = []
    la = []
    for l in self.layers:
      lx.append(a)
      z, a = l.forward(a)
      lz.append(z)
      la.append(a)
    return lx, lz, la

  def train(self, X, y):
    for epoch in range(self.epochs):
      batchsize = self.batchsize
      for b in range(0, len(X), batchsize):
        lx, lz, la = self.forward(X[b:min(b+batchsize, len(X))])
        da = mul(sub(la[-1], y[b:min(b+batchsize, len(X))]), 1)
        #print(nll(y[b:min(b+batchsize, len(X))], la[-1]))
        #print(la[-1])
        #print(epoch, sqrt(sum([sum(row)**2 for row in da])))
        #print(max(flatten(da)))
        #clf()
        """
        dimgrob(1, 320, 240, 0)
        plot(
          flatten(destandardize(X_tst, X_mu, X_sigma))[::shape(X)[1]],
          flatten(destandardize(self.forward(X_tst)[-1][-1], y_mu, y_sigma))
        )
        plot(
          flatten(destandardize(X_tst, X_mu, X_sigma))[::shape(X)[1]],
          flatten(destandardize(y_tst, y_mu, y_sigma))
        )
        blit(0,0,0,1)
        """
        #eval('wait')
        #show()
        for l in self.layers[::-1]:
          x = lx.pop(-1)
          z = lz.pop(-1)
          a = la.pop(-1)
          #print(shape(l.w))
          dw, db, da = l.backward(x, z, da)
          #print(shape(l.b), shape(db))
          l.w = sub(l.w, mul(self.alpha, dw))
          l.b = sub(l.b, mul(self.alpha, db))
          #print(dw)
      #print(epoch)

  def predict(self, X):
    return self.forward(X)[-1]

"""
X = shuffle([[n, cos(n), sin(n)] for n in arange(-2*pi, 2*pi, pi / 16)])
y = [[cos(x[0]) + 2 * sin(x[0]) * sin(x[0]) + x[0]] for x in X]
X_mu, X_sigma = mean_std(X)
X = standardize(X, X_mu, X_sigma)
y_mu, y_sigma = mean_std(y)
y = standardize(y, y_mu, y_sigma)
#print(axis(1, 2, 3, 4))
#plot(flatten(X), flatten(y))
#show()

index = int(.8 * len(X))

X_trn = X[:index]
y_trn = y[:index]
X_tst = X[index:]
y_tst = y[index:]

print(mean_std([[1],[2],[3],[4],[5]]))

model = NeuralNet()

print(model.layers[0].b)
model.layers[0].w = [[1, 2]]
model.layers[0].b = [[3, 4]]
model.layers[1].w = [[5], [6]]
model.layers[1].b = [[7]]

print(model.forward([[8]]))

lx, lz, la = model.forward(X_tst)
#print(apply(model.layers[0].activ.df, lz[0]))
#model.train(X_trn, y_trn)
"""