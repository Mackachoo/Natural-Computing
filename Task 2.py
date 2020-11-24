import numpy as np
import random as r

### Functions ----------------------------------------------------------------------------

def createInitials(N):
    initials = []
    for i in range(N):
        initials.append(np.random.randint(1,8,(r.randint(1,6))))
    return np.array(initials, dtype=object)

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

def scores(nws):
    nwSc = []
    for nw in nws:
        nwSc.append(r.randint(0,100))
    nwSc = np.array(nwSc)
    nwSc = np.cumsum(nwSc/np.sum(nwSc))
    return np.column_stack((nws,nwSc))


def selection(scored, pairs):
    selected = []
    print(scored)
    for i in range(2*pairs):
        rInt = r.random()
        for obj in scored:
            if obj[1] > rInt:
                selected.append(obj[0])
                break
    return np.array(selected, dtype=object)

### Constants ----------------------------------------------------------------------------

cRate = 0.5
mRate = 0.01
Iterations = 10
NumInitials = 100

### Program ------------------------------------------------------------------------------

networkSet = createInitials(NumInitials)
for i in range(Iterations):
    selected = selection(scores(networkSet))
    networkSet = []
    for pair in range(len(selected)//2):
        nwP1, nwP2 = crossover(selected[2*pair], selected[2*pair+1], cRate)
        nwP1 = mutate(nwP1, mRate)
        nwP2 = mutate(nwP2, mRate)

