import numpy as np
import random as r
import time as t
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split as testSplit
import warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm


### Functions ----------------------------------------------------------------------------

def createInitials(N):
    initials = []
    for _ in range(N):
        initials.append(r.sample(range(1,8),(r.randint(1,6))))
    return initials


def createTestSpirals(n, type='lin'):
    p = (720)*np.sqrt(np.random.rand(n//2,1))*(np.pi)/180
    posX = -p*np.cos(p) + np.random.rand(n//2,1)
    posY = p*np.sin(p) + np.random.rand(n//2,1)
    PosP = (posX, posY)
    PosN = (-posX, -posY)

    if 'sin' in type:
        sinPosX, sinPosY = np.sin(posX), np.sin(posY)
        PosP += (sinPosX, sinPosY)
        PosN += (-sinPosX, -sinPosY)       
    if 'squ' in type:
        squPosX, squPosY = posX**2, posY**2
        PosP += (squPosX, squPosY)
        PosN += (-squPosX, -squPosY)

    positions = np.vstack((np.hstack(PosP),np.hstack(PosN)))
    values = np.hstack((np.zeros(n//2),np.ones(n//2))).astype(int)
    return (positions, values)


def mutate(nw, mR):
    for i in range(len(nw)):
        if r.random() <= mR:
            if r.random() > 0.5:
                nw[i] += 1
            else:
                nw[i] -= 1
    #if r.random() <= mR and r.random() > 0.5:
    #    nw.append(1)
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


def scores(nws, type='lin', maxIt=3):
    nwSc = []
    posTrain, valueTrain = createTestSpirals(1000,type)
    posTest, valueTest = createTestSpirals(1000,type)
    for nw in nws:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')
            mlp = MLPClassifier(nw, max_iter=maxIt, solver='lbfgs', random_state=0, activation='tanh')
            mlp.fit(posTrain,valueTrain)
            nwSc.append(mlp.score(posTest, valueTest))
    return list(zip(nws,nwSc))


def selection(scored, pairs, elitism=True):
    sortScores = sorted(scored, key=lambda scored: scored[1])
    roulette = []
    for i in range(len(sortScores)):
        for _ in range(i):
            roulette.append(sortScores[i][0])
    selected = []
    if elitism:
        selected.append(roulette[-1])
    while len(selected) <= 2*pairs:
        selected.append(r.choice(roulette))
    return selected


def varianceCalc(scored):
    scoreSquares = [x[1]**2 for x in scored]
    return sum(scoreSquares)/len(scoreSquares)


### Main Program -------------------------------------------------------------------------

def main(crossoverRate=0.7, mutationRate=0.01, iterations=50, numInitials=50, survivalRate=0.5, elitism=True, maxMLPit=3, MLPtype='square'):
    log = "LOG FILE -------------------\n\n"
    networkSet = createInitials(numInitials)
    scoreList = []
    nwSets = [networkSet]

    for _ in tqdm(range(iterations)):
        scored = scores(networkSet, MLPtype, maxMLPit)
        scoreList.append([x[1] for x in scored])
        selected = selection(scored, int(len(networkSet)*survivalRate))
        networkSet = []
        for pair in range(len(selected)//2):
            for _ in range(int(0.5/survivalRate)):
                nwP1, nwP2 = crossover(selected[2*pair], selected[2*pair+1], crossoverRate)
                networkSet.append(mutate(nwP1, mutationRate))
                networkSet.append(mutate(nwP2, mutationRate))
        nwSets.append(networkSet)

    log += "\nOut lists -------\n\nNetwork list : "+str(nwSets)+"\n\nScores List : "+str(scoreList)+"\n"
    logFile = open("logFiles/log"+str(t.time())+"s.txt",'w')
    logFile.write(log)
    logFile.close()

    return {'finalNW':networkSet, 'nwSets':nwSets, 'scoreList':scoreList}

