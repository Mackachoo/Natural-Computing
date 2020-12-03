import numpy as np
import random as r
import time as t
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split as testSplit
import warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm


### Functions ----------------------------------------------------------------------------

def createInitials(N, dict):
    initials = []
    for _ in range(N):
        gpInput = {}
        for i in dict:
            gpInput[i] = r.choice(dict[i])
        initials.append([gpInput,r.sample(range(1,8),(r.randint(1,6)))])
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


def mutate(input, dict, mR):
    pr = input[0]
    nw = input[1]
    for i in dict:
        if r.random() <= mR:
            pr[i] = r.choice(dict[i])
    for i in range(len(nw)):
        if r.random() <= mR:
            if r.random() > 0.5:
                nw[i] += 1
            else:
                nw[i] -= 1
    if r.random() <= mR and r.random() > 0.5:
        nw.append(1)
    return [pr, [x for x in nw if x != 0]]

def crossover(input1, input2, cR):
    pr1 = input1[0]
    nw1 = input1[1]
    pr2 = input2[0]
    nw2 = input2[1]
    for i in pr1:
        if r.random() <= cR:
            pr1[i], pr2[i] = pr2[i], pr1[i]
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
    return [pr1, [x for x in nw1 if x != 0]], [pr2, [x for x in nw2 if x != 0]]

def scores(nws, maxIt, type='lin'):
    nwSc = []
    posTrain, valueTrain = createTestSpirals(1000,type)
    posTest, valueTest = createTestSpirals(1000,type)
    for nw in nws:
        shape = nw[1]
        activ = 'tanh'
        alpha = 0.0001
        if 'activation' in nw[0]:
            activ = nw[0]['activation']
        #if 'maxIter' in nw[0]:
        #    maxIt = nw[0]['maxIter']
        #if 'alpha' in nw[0]:
        #    alpha = nw[0]['alpha']
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')
            mlp = MLPClassifier(shape, max_iter=maxIt, solver='adam', random_state=0, activation=activ, learning_rate_init=alpha,)
            mlp.fit(posTrain,valueTrain)
            nwSc.append(mlp.score(posTest, valueTest))
            #print(nw," score - ",mlp.score(posTest, valueTest))
    return list(zip(nws,nwSc))


def selection(scored, pairs, elitism):
    sortScores = sorted(scored, key=lambda scored: scored[1])
    roulette = []
    for i in range(len(sortScores)):
        for _ in range(i):
            roulette.append(sortScores[i][0])
    selected = []
    if elitism:
        selected.append(roulette[-1])
    while len(selected) < 2*pairs:
        selected.append(r.choice(roulette))
    return selected

def oldSelection(scored, pairs, elitism):
    sortScores = sorted(scored, key=lambda scored: scored[1])
    nws = [x[0] for x in sortScores]
    val = np.cumsum(np.array([x[1] for x in sortScores]))
    selected = []
    if elitism:
        selected.append(sortScores[-1][0])
    while len(selected) < 2*pairs:
        rInt = r.random()*sortScores[-1][1]
        for i in range(len(nws)):
            if rInt < val[i]:
                selected.append(nws[i])
                break
    return selected


def varianceCalc(scored):
    scoreSquares = [x[1]**2 for x in scored]
    return sum(scoreSquares)/len(scoreSquares)


### Main Program -------------------------------------------------------------------------

def main(crossoverRate=0.7, mutationRate=0.01, iterations=50, numInitials=100, survivalRate=0.5, elitism=True, maxMLPit=2, MLPtype='square',  oldSelect=False):
    logger = "LOG FILE -------------------\n\n"
    logFile = open("logFiles/log"+str(123)+"s.txt",'a')

    gpDict = {'activation':['identity','logistic','tanh','relu']}
#    gpDict = {'activation':['identity','logistic','tanh','relu'], 'alpha':[1.0,0.1,0.01,0.001,0.0001], 'iterations':[1,2,3,4], 'type':['lin','squ','sin','sinsqu']}
    networkSet = createInitials(numInitials, gpDict)
    scoreList = []
    nwSets = [networkSet]

    for _ in tqdm(range(iterations)):
        scored = scores(networkSet, maxMLPit, MLPtype)
        scoreList.append([x[1] for x in scored])
        if oldSelect:
            selected = oldSelection(scored, int(len(networkSet)*survivalRate), elitism)
        else:
            selected = selection(scored, int(len(networkSet)*survivalRate), elitism)
        networkSet = []
        for pair in range(len(selected)//2):
            for _ in range(int(0.5/survivalRate)):
                nwP1, nwP2 = crossover(selected[2*pair], selected[2*pair+1], crossoverRate)
                networkSet.append(mutate(nwP1, gpDict, mutationRate))
                networkSet.append(mutate(nwP2, gpDict, mutationRate))
        nwSets.append(networkSet)

    logger+= "\nOut lists -------\n\nNetwork list ~: "+str(nwSets)+"\n\nScores List : "+str(scoreList)+"\n"
    logFile.write(logger)
    logFile.close()

    return {'nwSets':nwSets, 'scoreList':scoreList}

#run = main(oldSelect=True, numInitials=100)
#print(run['nwSets'])
#print(run['scoreList'])

