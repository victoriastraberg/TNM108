import numpy
import time

def sum_trad():
  start = time.time()
  X = range(10000000)
  Y = range(10000000)
  Z = []
  for i in range(len(X)):
      Z.append(X[i] + Y[i])
  return time.time() - start  

#Not using Numpy
def sum_numpy():
  start = time.time()
  X = numpy.arange(10000000)
  Y = numpy.arange(10000000)
  Z = X+Y
  return time.time() - start
print ('time sum:', sum_trad(),' time sum_numpy:', sum_numpy())
