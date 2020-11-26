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
    

#def scores(nws, data, type='linear'):
#    nwSc = []
#    if type == 'square':
#        pos = np.concatenate((data[:,0:2],data[:,0:2]**2), axis=1)
#    elif type == 'sin':
#        pos = np.concatenate((data[:,0:2],np.sin(data[:,0:2])), axis=1)
#    else:
#        pos = data[:,0:2]
#    print(pos)
#    posTrain, posTest, valueTrain, valueTest = testSplit(pos,data[:,2], test_size=0.5)
#    posTrain, posTest, valueTrain, valueTest = np.array(posTrain), np.array(posTest), np.array([int(x) for x in valueTrain]), np.array([int(x) for x in valueTest])    
#    for nw in nws:
#        with warnings.catch_warnings():
#            warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')
#            mlp = MLPClassifier(nw, solver='lbfgs', random_state=0, activation='tanh')
#            mlp.fit(posTrain,valueTrain)
#            #print(nw, mlp.score(posTest, valueTest))
#            nwSc.append(mlp.score(posTest, valueTest))
#    #print("End-------")
#    return list(zip(nws,nwSc))
#
#def createInitials(N):
#    initials = []
#    for _ in range(N):
#        initials.append(r.sample(range(1,8),(r.randint(1,6))))
#    return initials
#
#
#nws = createInitials(10)
#data = np.loadtxt(open("two_spirals.dat"))
#nws = [[8,0,8,8]]
#
#print(np.array(scores(nws,data,type='square')))