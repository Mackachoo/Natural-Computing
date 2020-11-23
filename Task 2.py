import numpy as np
import random as r

cRate = 0.5
mRate = 0.01


def createInitials(N):
    initials = []
    for i in range(N):
        initials.append(np.random.randint(1,8,(r.randint(1,6))))
    return np.array(initials)


def mutate(nw, mR):
    for i in range(len(nw)):
        if r.random() <= mR:
            if r.random() > 0.5:
                nw[i] += 1
            else:
                nw[i] -= 1
    if r.random() <= mR:
        nw = np.append(nw,[1])
    return nw[nw != 0]

def crossover(nw1, nw2, cR):
    for i in range(max(len(nw1),len(nw2))):
        if r.random() <= cR:
            print(i)
            if len(nw1) <= i:
                nw1 = np.append(nw1,nw2[i])
                nw2[i] = 0
            elif len(nw2) <= i:
                nw2 = np.append(nw2,nw1[i])
                nw1[i] = 0
            else:
                nw1[i], nw2[i] = nw2[i], nw1[i]
    return nw1[nw1 != 0], nw2[nw2 != 0]

print(crossover(np.array([1,2,3]), np.array([9,8,7,6]), 0.3))



#np.random.randint(1,8,(1,r.randint(1,6)))