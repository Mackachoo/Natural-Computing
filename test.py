import numpy as np
import time as t
import random as r

a = np.random.randint(0,4,(999,10))
b = a.tolist()

#print(a)
#print(b)

start = t.time()
for i in range(999):
    x = a[i][a[i] != 0]
print(t.time()-start)

start = t.time()
for i in range(999):
    x  = [e for e in b[i] if e != 0]
print(t.time()-start)
#print(a)
