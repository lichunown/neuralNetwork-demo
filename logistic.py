import numpy as np
import matplotlib.pyplot as plt


data = [
        [1,2,1],
        [2,3,1],
        [0.5,2,1],
        [3,5,1],
        [3,4,1],
        [2,4,1],
        
        [2,1,0],
        [3,2,0],
        [2,0.5,0],
        [5,3,0],
        [4,3,0],
        [4,2,0],
    ]

data = np.array(data)

theta = np.array([
            [1],
            [5],
            [1],
        ],'float'
    )


x = np.concatenate((np.ones([data.shape[0],1]),data[:,[0,1]]/5),1)
y = np.array([data[:,2]]).T


plt.plot(data[np.where(data[:,2]==1),0],data[np.where(data[:,2]==1),1],'r+')
plt.plot(data[np.where(data[:,2]==0),0],data[np.where(data[:,2]==0),1],'b*')


def logistic(x):
    return 1/(1+np.exp(-x))

def hFunction(theta,x):
    return logistic(np.dot(x,theta))

def costFunction(theta,x,m=1):
    #cost[np.where(data[:,2]==1),0] = -np.log(h[np.where(data[:,2]==1),0])
    #cost[np.where(data[:,2]==0),0] = -np.log(1-h[np.where(data[:,2]==0),0])
    cost = (1/m)*np.sum(-y*np.log(hFunction(theta,x))-(1-y)*np.log(1-hFunction(theta,x)))
    dJ = np.array([(1/m)*np.sum(np.dot(x.T,hFunction(theta,x)-y),1)]).T
    return cost,dJ

def train(x=x,y=y,theta=theta,alpha=0.01):
    while True:
        cost,dJ = costFunction(theta,x)
        theta -= dJ*alpha
        if np.sum(dJ)<0.001:
            break
    return theta,cost

def plotLine(theta=theta,data=data):
    fx = lambda x:-theta[1]/theta[2]*x-theta[0]/theta[2]
    x = np.arange(0,5,0.1)
    y = fx(x)
    plt.hold(True)
    plt.plot(data[np.where(data[:,2]==1),0],data[np.where(data[:,2]==1),1],'r+')
    plt.plot(data[np.where(data[:,2]==0),0],data[np.where(data[:,2]==0),1],'b*')
    plt.plot(x,y)

train()
plotLine()