from linalg import *

from mnistdata import *
from activation import *

class SoftmaxRegression:
  def __init__(self, alpha=.05, batchsize=32, epochs=50, seed=0):
    self.alpha = alpha
    self.batchsize = batchsize
    self.epochs = epochs

    self.w = None
    self.trn_loss = None
    self.vld_loss = None

  def fit(self, X, y, X_vld=None, y_vld=None, init=True):
    self.trn_loss = []
    self.vld_loss = []
    if init:
      self.w = rand(shape(X)[1], shape(y)[1])
    for epoch in range(self.epochs):
      for b in range(1, len(X), self.batchsize):
        xb = X[b:min(b + self.batchsize, len(X))]
        yb = y[b:min(b + self.batchsize, len(y))]
        z = dot(xb, self.w)
        y_probs = Softmax.f(z)
        self.w = sub(self.w, mul(dot(transpose(xb), sub(y_probs, yb)), self.alpha / len(xb)))
      y_hat_trn = Softmax.f(dot(X, self.w))
      #print(Softmax.f(dot(X, self.w)))
      print(epoch, nll(y, y_hat_trn))
      #print()

  def predict(self, X):
    return [argmax(sample) for sample in Softmax.f(dot(X, self.w))]

#X = readsamples