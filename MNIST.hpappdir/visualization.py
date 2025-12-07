from urandom import *

from mnistdata import *

def show(X, labels):
  fillrect(0, 0, 0, 320, 240, 0, 0)
  #for s in X:
  for i, (s, l) in enumerate(zip(X, labels)):
    ix = i % 10
    iy = i // 10
    for y in range(28):
      for x in range(28):
        c = 0x10101 * s[28 * y + x]
        #fillrect(0, 8 * x, 8 * y, 8, 8, c, c)
        pixon(0, 32 * ix + x + 4, 28 * iy + y, c)
    textout(0, 32 * ix, 28 * iy, str(l.index(1)) + ':', 0xffffff)
    #print(i, len(X), len(y))

X, y = readsamples(80)
show(X[:80], y[:80])
eval('wait')