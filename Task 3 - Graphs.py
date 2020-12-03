import matplotlib.pyplot as plt
from celluloid import Camera
import numpy as np
from tqdm import tqdm
import random as r
task = __import__("Task 3")


### Graphs --------------------------------------------------------------------------------

def Graphs3D(nwSets=None, scoreList=None, type=0, snap=False):
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
                    ax.set_zlabel('Layer Width')
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


def Graphs2D(scoreList=None, type=0):
    if type == 0:
        x = np.arange(len(scoreList))
        plt.ylim(0,1)
        plt.xlabel("Iterations")
        plt.ylabel("Average Score")
        plt.plot(x, [sum(x)/len(x) for x in scoreList])
    else:
        print("Invalid Type")
    plt.show()

for i in range(100):
    print()
    outDefault = task.main(numInitials=10)
    for i in range(3):
        print(outDefault['nwSets'][len(outDefault['nwSets'])-(i+1)][0]," : ",outDefault['scoreList'][len(outDefault['scoreList'])-(i+1)][0])
    #Graphs2D(scoreList=outDefault['scoreList'])
       


#Graphs3D(nwSets=outDefault['nwSets'], scoreList=outDefault['scoreList'], type=2, snap=True)

#fig = plt.figure(figsize=(10,5))
#ax = fig.add_subplot(211)
#bx = fig.add_subplot(212)
#outLin = task.main(survivalRate=0.125, numInitials=1000, elitism=True)
#outSin = task.main(survivalRate=0.125, numInitials=1000, elitism=False)
#outSqu = task.main(survivalRate=0.125, numInitials=10, elitism=True)
#outSqS = task.main(survivalRate=0.125, numInitials=10, elitism=False)
#
#iteration = np.arange(50)
#ax.set_ylim(0,1.01)
#bx.set_ylim(0,1.01)
#ax.set_ylabel('Score')
#ax.set_xlabel('Iteration')
#ax.plot(iteration, [sum(x)/len(x) for x in outLin['scoreList']], label="Elitism")
#ax.plot(iteration, [sum(x)/len(x) for x in outSin['scoreList']], label="No Elitism")
#bx.plot(iteration, [sum(x)/len(x) for x in outSqu['scoreList']], label="Elitism")
#bx.plot(iteration, [sum(x)/len(x) for x in outSqS['scoreList']], label="No Elitism")
#ax.legend(loc='best')
#bx.legend(loc='best')
#ax.set_title('N=1000')
#bx.set_title('N=10')
#plt.savefig('figures/Elitismx2.png', transparent=True)
#plt.show()