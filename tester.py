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


##nws = createInitials(10)
#for i in range(20):
#    nws = [r.sample(range(1,8),(r.randint(1,7)))]
#    print(np.array(scores(nws, type='square')))
#
#
#outDefault = task.main()
##Graphs3D(nwSets=outDefault['nwSets'], scoreList=outDefault['scoreList'], type=1, snap=True)
##Graphs2D(scoreList=outDefault['scoreList'])
#
#fig = plt.figure(figsize=(30,15))
#ax = fig.add_subplot(111)
##outLin = task.main(MLPtype='linear')
##outSin = task.main(MLPtype='square')
##outSqu = task.main(MLPtype='sineee')
##outSqS = task.main(MLPtype='sinsqu')
#
#iteration = np.arange(50)
#ax.set_ylim(0,1.01)
#ax.set_ylabel('Score')
#ax.set_xlabel('Iteration')
#ax.legend(loc='best')
#ax.plot(iteration, [sum(x)/len(x) for x in outDefault['scoreList']], label="Avg")
#ax.plot(iteration, [max(x) for x in outDefault['scoreList']], label="Max")
#ax.plot(iteration, [min(x) for x in outDefault['scoreList']], label="Min")
##ax.plot(iteration, [sum(x)/len(x) for x in outSqS['scoreList']], label="sqs")
#plt.savefig('figures/Average Scores 1000.png', transparent=True)
#plt.show()

gpDict = {'activation':['i','l','t','r'], 'alpha':[1.0,0.1,0.01,0.001,0.0001], 'iterations':[1,2,3,4], 'type':['lin','squ','sin','sinsqu']}

print(len(gpDict))
for i in gpDict:
    print(i)

abra = ['r',1,2,'lin',2,3,1]
print(abra[0:])