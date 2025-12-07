from hpprime import *

from mnistdata import *
from lr import *
from nn import *
from utils import *
from activation import *

n = 100
model_sr = SoftmaxRegression(
    epochs=10,
    alpha=0.05,
)
model_nn = NeuralNet(
  alpha=.1,
  epochs=50,
  sizes=None
)

model_nn.layers.append(Perceptron(784, 64, ReLU))
model_nn.layers.append(Perceptron(64, 64, ReLU))
model_nn.layers.append(Perceptron(64, 32, Linear))
model_nn.layers.append(Perceptron(32, 10, Softmax))

for i in range(0, 200, n):
  X_trn, y_trn, X_vld, y_vld = get_preprocessed(n, offset=i)
  print(i)
  #model_sr.fit(X_trn, y_trn, X_vld, y_vld, init=(i==0))
  model_nn.train(X_trn, y_trn)

screen = [0]*784
dimgrob(1, 320, 240, 0)
while True:
  if mouse()[0]:
    mx, my = mouse()[0]
    if mx < 224 and my < 224:
      sx = mx // 8
      sy = my // 8
      screen[28 * sy + sx] = 1
      fillrect(1, 8 * sx, 8 * sy, 8, 8, 0xffffff, 0xffffff)
  blit(0, 0, 0, 1)

  if keyboard()>>30&1:
    #print(model_)
    print((model_nn.forward([screen])[-1][-1][0]))
    #print(model_sr.predict([screen]))
    eval('wait')