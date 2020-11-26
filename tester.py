import numpy as np
import random as r
import time as t
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split as testSplit
import warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
import matplotlib.pyplot as plt
from celluloid import Camera


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
    

def scores(nws, type='lin', maxIt=1):
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

def createInitials(N):
    initials = []
    for _ in range(N):
        initials.append(r.sample(range(1,8),(r.randint(1,6))))
    return initials


#nws = createInitials(10)
for i in range(20):
    nws = [r.sample(range(1,8),(r.randint(1,7)))]
    print(np.array(scores(nws, type='square')))