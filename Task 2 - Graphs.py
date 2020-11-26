import matplotlib.pyplot as plt
from celluloid import Camera
import numpy as np
from tqdm import tqdm
task = __import__("Task 2")


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

output = task.main()

Graphs2D(output['scoreList'])
Graphs3D(nwSets=output['nwSets'], type=1, snap=False)
