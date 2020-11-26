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


def scores(nws, data, type='lin'):
    nwSc = []
    posTrain, valueTrain = createTestSpirals(1000,type)
    posTest, valueTest = createTestSpirals(1000,type)
    for nw in nws:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')
            mlp = MLPClassifier(nw, max_iter=3, solver='lbfgs', random_state=0, activation='tanh')
            mlp.fit(posTrain,valueTrain)
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
    return selected


def varianceCalc(scored):
    scoreSquares = [x[1]**2 for x in scored]
    return sum(scoreSquares)/len(scoreSquares)


### Constants ----------------------------------------------------------------------------

crossoverRate = 0.7
mutationRate = 0.01
iterations = 50
numInitials = 50
survivalRate = 0.5
variance = 0.001
data = np.loadtxt(open("two_spirals.dat"))


### Output -------------------------------------------------------------------------------

log = "LOG FILE -------------------\n\n\n"
networkSet = createInitials(numInitials)
varList = []
scoreList = []
nwSets = [networkSet]
#print(networkSet)


### Program ------------------------------------------------------------------------------

for _ in tqdm(range(iterations)):
    scored = scores(networkSet, data, 'square')
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
print('\nResults:',networkSet, '\n')

log += "\nOut lists -------\n\nNetwork list : "+str(nwSets)+"\n\nScores List : "+str(scoreList)+"\n"
logFile = open("logFiles/log"+str(t.time())+"s.txt",'w')
logFile.write(log)
logFile.close()


### Graphs --------------------------------------------------------------------------------

def Graphs3D(type=0, snap=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    cam = Camera(fig)
    size = 10
    angle = 0

    if type == 2:                   # Score Line
       ax.view_init(30, 120)
       ax.set_xlim3d(0, len(scoreList))
       ax.set_ylim3d(0, len(scoreList[0]))
       for xS in range(len(scoreList)):
           for yS in range(len(scoreList[xS])):
               z = [x[yS] for x in scoreList[:xS+1]]
               #print(len(x[:xS+1]), len(y[:xS+1]), len(z))
               ax.plot(np.arange(xS+1), np.ones(xS+1)*yS, z)
           if xS != 0:
               plt.pause(0.1)
               pass
           cam.snap()
    else:                           # Surface and Bar Chart
        for nw in tqdm(nwSets):
            ax.view_init(30, 80)
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
                    ax.bar(x,n,count, zdir='x')
                    #plt.pause(0.01)
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
            anim.save("imageOut/anim"+str(r.randint(0,100))+".gif", writer='Pillow', fps=10, runTotal=len(nwSets))
        except:
            anim.save("imageOut/anim"+str(r.randint(0,100))+".gif", writer='Pillow', fps=10)
    plt.show()


def Graphs2D(type=0):
    if type == 0:
        x = np.arange(len(varList))
        plt.xlabel("Iterations")
        plt.ylabel("Variance")
        plt.plot(x, varList)
    elif type == 1:
        x = np.arange(len(varList))
        plt.ylim(0,1)
        plt.xlabel("Iterations")
        plt.ylabel("Average Score")
        plt.plot(x, [sum(x)/len(x) for x in scoreList])
    else:
        print("Invalid Type")
    plt.show()

Graphs2D(1)
#Graphs3D(1, True)
#Graphs3D(2, False)
#print(np.array(scoreList))
