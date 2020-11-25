import numpy as np
import random as r
from time import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split as testSplit
import warnings
from sklearn.exceptions import ConvergenceWarning
from tqdm import tqdm
import matplotlib.pyplot as plt
from celluloid import Camera


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
    else:
        pos = data[:,0:2]
    posTrain, posTest, valueTrain, valueTest = testSplit(pos,data[:,2], test_size=0.5)
    posTrain, posTest, valueTrain, valueTest = np.array(posTrain), np.array(posTest), np.array([int(x) for x in valueTrain]), np.array([int(x) for x in valueTest])    
    for nw in nws:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')
            mlp = MLPClassifier(nw, max_iter=3, solver='lbfgs', random_state=0, activation='tanh')
            mlp.fit(posTrain,valueTrain)
            #print(nw, mlp.score(posTest, valueTest))
            nwSc.append(mlp.score(posTest, valueTest))
    #print("End-------")
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
    #print(selected, pairs,'select')
    return selected


def varianceCalc(scored):
    scoreSquares = [x[1]**2 for x in scored]
    return sum(scoreSquares)/len(scoreSquares)


### Constants ----------------------------------------------------------------------------

crossoverRate = 0.5
mutationRate = 0.01
iterations = 100
numInitials = 8
survivalRate = 0.25
variance = 0.001
data = np.loadtxt(open("two_spirals.dat"))


### Output -------------------------------------------------------------------------------

networkSet = createInitials(numInitials)
varList = []
scoreList = []
nwSets = [networkSet]
#print(networkSet)


### Program ------------------------------------------------------------------------------

for _ in tqdm(range(iterations)):
    scored = scores(networkSet, data, 'sin')
    scoreList.append([x[1] for x in scored])
    curVar = varianceCalc(scored)
    varList.append(curVar)
    if curVar < variance:
        print("Converged", curVar)
        break
    selected = selection(scored, int(len(networkSet)*survivalRate))
    networkSet = []
    for pair in range(len(selected)//2):
        for _ in range(int(0.5/survivalRate)):
            nwP1, nwP2 = crossover(selected[2*pair], selected[2*pair+1], crossoverRate)
            networkSet.append(mutate(nwP1, mutationRate))
            networkSet.append(mutate(nwP2, mutationRate))
    nwSets.append(networkSet)
print('\n',networkSet)


### Graphs --------------------------------------------------------------------------------

def Graphs3D(type=0, snap=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cam = Camera(fig)
    size = 10
    angle = 0

    if type == 2:
        ax.view_init(30, 120)
        ax.set_xlim3d(0, len(scoreList))
        ax.set_ylim3d(0, len(scoreList[0]))
        ax.set_zlim3d(0, )
        for xS in range(len(scoreList)):
            for yS in range(len(scoreList[xS])):
                z = [x[yS] for x in scoreList[:xS+1]]
                #print(len(x[:xS+1]), len(y[:xS+1]), len(z))
                ax.plot(np.arange(xS+1), np.ones(xS+1)*yS, z)
            if xS != 0:
                plt.pause(0.1)
                pass
            cam.snap()
    else:
        for nw in tqdm(nwSets):
            ax.view_init(30, 120)
            if type == 0:
                X = np.arange(0, size)
                Y = np.arange(0, size)
                X, Y = np.meshgrid(X, Y)
                Z = np.zeros((size,size))
                for x in range(size):
                    for y in range(size):
                        try:
                            Z[x,y] = nw[x][y]
                        except:
                            pass
                ax.plot_surface(X, Y, Z)
            elif type == 1:
                ax.view_init(30, 60)
                count = 0
                for n in nw:
                    count += 1
                    x = np.arange(0, len(n))
                    ax.set_xlabel('Population')
                    ax.set_ylabel('Layer Number')
                    ax.set_zlabel('Layer Depth')
                    ax.bar(n,x,count, zdir='x')
                    plt.pause(0.01)
            else:
                print("Invalid Type")
                break
            angle += 1
            #ax.view_init(30, 0)
            #plt.pause(0.1)
            cam.snap()

    if snap:
        anim = cam.animate()
        try:
            anim.save('new.gif', writer='Pillow', fps=10, runTotal=len(nwSets))
        except:
            anim.save('new.gif', writer='Pillow', fps=10)
    plt.show()


def Graphs2D(type=0):
    if type == 0:
        x = np.arange(len(varList))
        plt.xlabel("Iterations")
        plt.ylabel("Variance")
        plt.plot(x, varList)
    else:
        print("Invalid Type")
    plt.show()

#Graphs2D()
Graphs3D(0, True)
#print(np.array(scoreList))
