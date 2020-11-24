import numpy as np
import random as r
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split as testSplit


### Functions ----------------------------------------------------------------------------

def createInitials(N):
    initials = []
    for _ in range(N):
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
            if len(nw1) <= i:
                nw1.append(nw2[i])
                nw2[i] = 0
            elif len(nw2) <= i:
                nw2.append(nw1[i])
                nw1[i] = 0
            else:
                nw1[i], nw2[i] = nw2[i], nw1[i]
    return [x for x in nw1 if x != 0], [x for x in nw2 if x != 0]


def scores(nws, data, type='linear'):
    nwSc = []
    if type == 'square':
        pos = np.concatenate((data[:,0:2],data[:,0:2]**2), axis=1)
    elif type == 'sin':
        pos = np.concatenate((data[:,0:2],np.sin(data[:,0:2])), axis=1)
        print(pos)
    else:
        pos = data[:,0:2]
    posTrain, posTest, valueTrain, valueTest = testSplit(pos,data[:,2], test_size=0.5)
    posTrain, posTest, valueTrain, valueTest = np.array(posTrain), np.array(posTest), np.array([int(x) for x in valueTrain]), np.array([int(x) for x in valueTest])
    for nw in nws:
        mlp = MLPClassifier(nw, max_iter=3, solver='lbfgs', random_state=0, activation='tanh')
        mlp.fit(posTrain,valueTrain)
        nwSc.append(mlp.score(posTest, valueTest))
    nwSc = np.array(nwSc)
    nwSc = list(np.cumsum(nwSc/np.sum(nwSc)))
    return list(zip(nws,nwSc))


def selection(scored, pairs, elitism=True):
    selected = []
    if elitism:
        selected.append(max(scored)[0])
    while len(selected) <= 2*pairs:
        rInt = r.random()
        for obj in scored:
            if obj[1] > rInt:
                selected.append(obj[0])
                break
    return selected


def converged(scored, variance):
    scoreSquares = [x[1]**2 for x in scored]
    return sum(scoreSquares)/len(scoreSquares) < variance


### Constants ----------------------------------------------------------------------------

crossoverRate = 0.5
mutationRate = 0.01
iterations = 10
numInitials = 100
survivalRate = 0.25
variance = 0.3
data = np.loadtxt(open("two_spirals.dat"))

### Program ------------------------------------------------------------------------------

networkSet = createInitials(numInitials)
for _ in range(iterations):
    scored = scores(networkSet, data, 'sin')
    if converged(scored, variance):
        print("Converged")
        break
    selected = selection(scored, int(len(networkSet)*survivalRate))
    networkSet = []
    for pair in range(len(selected)//2):
        for _ in range(int(0.5/survivalRate)):
            nwP1, nwP2 = crossover(selected[2*pair], selected[2*pair+1], crossoverRate)
            networkSet.append(mutate(nwP1, mutationRate))
            networkSet.append(mutate(nwP2, mutationRate))

print(networkSet)
