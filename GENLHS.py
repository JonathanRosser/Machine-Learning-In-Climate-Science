import GPy
import numpy as np
from GPy.models.gp_regression import GPRegression as reg
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.integrate import odeint
from pyDOE import *



def yfromx(x,Num):
    y=np.zeros((Num,N))
    for j in range(0,Num):
        print(j)
        x0=np.random.rand(K*(J+1))
        y[j,:]=output(odeint(Lorenz96,x0,t,tuple(x[j,:])),fasttime,fastspin)
    return y
#Running average function
def runavg(func,x,W):
    #Function is the function which will be calculated over each of the windows, i.e. a mean or variance etc
    #x is the array over which the function will be calculated
    #W is the window length
    print(func)
    if func==np.cov:
        y=np.zeros((len(x)-W,K*(J+1),K*(J+1)))
        for i in range(0,len(x)-W):
            y[i,:,:]=np.cov(x[i:W+i],rowvar=False)
        return y
    else:
        y=np.zeros(len(x)-W)
        for i in range(0,len(x)-W):
            y[i]=func(x[i:W+i])
        return y

#Function to calculate the XY term which will allow for correlation between X and Y which might be significant
def XY(x,time,spin):
    y=np.zeros((time-spin,K))
    for i in range(0,K):
        y[:,i]=np.mean(x[spin:,K+i*J:K+(i+1)*J],axis=1)
    XY=np.mean(np.multiply(x[spin:,:K],y),axis=1)
    return XY


def XY2(x,time,spin):
    y=np.zeros((time-spin,K))
    for i in range(0,K):
        y[:,i]=np.mean(x[spin:,K+i*J:K+(i+1)*J],axis=1)
    XY=np.mean(np.multiply(x[spin:,:K],y))
    return XY

def output(x,time,spin):
    y=np.zeros(5)
    y[0]=np.mean(x[spin:,:K]) #Mean of the slow variables
    y[1]=np.mean(x[spin:,K:]) #Mean of the fast variables
    y[2]=np.mean(np.square(x[spin:,:K])) #Mean of the squared slow variables
    y[3]=np.mean(np.square(x[spin:,K:])) #Mean of the squared fast variables
    y[4]=XY2(x,time,spin) #Mean of XY
    return y

def Lorenz96(x,t,h,F,c,b):
    #Create vector for the derivatives
    d=np.zeros(K*(J+1))
    #Slow variable case
    #Consider the edge cases first (i=1,2,K):
    d[0]=-x[K-1]*(x[K-2]-x[1])-x[0]+F-h*c*np.mean(x[K:K+J])
    d[1]=-x[0]*(x[K-1]-x[2])-x[1]+F-h*c*np.mean(x[K+J:K+2*J])
    d[K-1]=-x[K-2]*(x[K-3]-x[0])-x[K-1]+F-h*c*np.mean(x[K+J*(K-1):K+J*(K)])
    #General case:
    for i in range(2,K-1):
        d[i]=-x[i-1]*(x[i-2]-x[i+1])-x[i]+F-h*c*np.mean(x[K+J*i:K+J*(i+1)])

#Fast variable case
    for l in range(0,K):
        #Consider the edge cases first (i=1,J-1,J):
        N=K+l*J
        d[N]=c*(-b*x[N+1]*(x[N+2]-x[N+J-1])-x[N]+h*x[l]/J)
        d[N+J-1]=c*(-b*x[N]*(x[N+1]-x[N+J-2])-x[N+J-1]+h*x[l]/J)
        d[N+J-2]=c*(-b*x[N+J-1]*(x[N]-x[N+J-3])-x[N+J-2]+h*x[l]/J)
    #General case:
        for i in range(1,J-2):
            N=K+l*J+i
            d[N]=c*(-b*x[N+1]*(x[N+2]-x[N-1])-x[N]+h*x[l]/J)
    return d



X=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/ENKIparameters400x4sample2.npy")
MEAN=np.zeros((4))
STD=np.zeros((4))
MEAN=np.mean(X[:,1200:],axis=1)
STD=np.std(X[:,1200:],axis=1)
#STD=(0.1,1,0.1,1)

Ntrain=500
Ntest=20
M=4
N=5
K=36
J=10
fasttime=1000
fastspin=100
t=np.arange(0.0,fasttime/100,0.01)



for k in range(0,2):
    lgdtrain=lhs(M,samples=Ntrain,criterion='maximin')
    xlhstrain=np.zeros((Ntrain,M))
    xlhstrain[:,0]=lgdtrain[:,0]*STD[0]*2-STD[0]+MEAN[0]
    xlhstrain[:,1]=lgdtrain[:,1]*STD[1]*2-STD[1]+MEAN[1]
    xlhstrain[:,2]=lgdtrain[:,2]*STD[2]*2-STD[2]+MEAN[2]
    xlhstrain[:,3]=lgdtrain[:,3]*STD[3]*2-STD[3]+MEAN[3]
    ylhstrain=yfromx(xlhstrain,Ntrain)
    np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/LHSENKIMSXTrain{}.npy".format(k),xlhstrain)
    np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/LHSENKIMSYTrain{}.npy".format(k),ylhstrain)







    #lgdtest=lhs(4,samples=Ntest,criterion='maximin')
    #xlhstest=np.zeros((Ntest,M))
    #xlhstest[:,0]=lgdtest[:,0]*6-3
    #xlhstest[:,1]=lgdtest[:,1]*52-16
    #xlhstest[:,2]=lgdtest[:,2]*5+5
    #xlhstest[:,3]=lgdtest[:,3]*52-21
    #ylhstest=yfromx(xlhstest,Ntest)
    #np.save("/home/jonathan/Documents/WorkPlacements/Caltech/Data/LHSXTest{}.npy".format(k),xlhstest)
    #np.save("/home/jonathan/Documents/WorkPlacements/Caltech/Data/LHSYTest{}.npy".format(k),ylhstest)













