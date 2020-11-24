import numpy as np
import random as r

### Functions ----------------------------------------------------------------------------

def createInitials(N):
    initials = []
    for i in range(N):
        initials.append(r.sample(range(1,8),(r.randint(1,6))))
    return initials

    
def mutate(nw, mR):
    for i in range(len(nw)):
        if r.random() <= mR:
            if r.random() > 0.5:
                nw[i] += 1
            else:
                nw[i] -= 1
    if r.random() <= mR:
        nw.append(1)
    return [x for x in nw if x != 0]


def crossover(nw1, nw2, cR):
    for i in range(max(len(nw1),len(nw2))):
        if r.random() <= cR:
            print(i)
            if len(nw1) <= i:
                nw1.append(nw2[i])
                nw2[i] = 0
            elif len(nw2) <= i:
                nw2.append(nw1[i])
                nw1[i] = 0
            else:
                nw1[i], nw2[i] = nw2[i], nw1[i]
    return [x for x in nw1 if x != 0], [x for x in nw2 if x != 0]


def scores(nws):
    nwSc = []
    for nw in nws:
        nwSc.append(r.randint(0,100))
    nwSc = np.array(nwSc)
    nwSc = list(np.cumsum(nwSc/np.sum(nwSc)))
    return dict(nws,nwSc)


def selection(scored, pairs, elitism =True):
    selected = []
    if elitism:
        selected.append(max(scored))
    while len(selected) <= 2*pairs:
        rInt = r.random()
        for nw in scored:
            if scored[nw] > rInt:
                selected.append(nw)
                break
    return selected


### Constants ----------------------------------------------------------------------------

crossoverRate = 0.5
mutationRate = 0.01
iterations = 10
numInitials = 100
survivalRate = 0.25

### Program ------------------------------------------------------------------------------

networkSet = createInitials(numInitials)
for i in range(iterations):
    selected = selection(scores(networkSet), int(len(networkSet)*survivalRate))
    networkSet = []
    for pair in range(len(selected)//2):
        nwP1, nwP2 = crossover(selected[2*pair], selected[2*pair+1], crossoverRate)
        networkSet.append(mutate(nwP1, mutationRate))
        networkSet.append(mutate(nwP2, mutationRate))

print(networkSet)