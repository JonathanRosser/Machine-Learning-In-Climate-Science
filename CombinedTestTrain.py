import GPy
import numpy as np
from GPy.models.gp_regression import GPRegression as reg
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.integrate import odeint
from pyDOE import *
import time

###################Define Useful functions########################


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

def gpmake(X,Y,name):
    print("Making {}".format(name))
    kernel=GPy.kern.RBF(M)
    noise=0.1
    normalizer=False
    gp=reg(X,Y,kernel,noise_var=noise,normalizer=normalizer)
    gp.optimize()
    Models[name]=gp

def yfromx(x,Num):
    y=np.zeros((Num,N))
    print("start at:")
    print(time.ctime(),time.time())
    for j in range(0,Num):
        print(j)
        x0=np.random.rand(K*(J+1))
        y[j,:]=output(odeint(Lorenz96,x0,t,tuple(x[j,:])),fasttime,fastspin)
    print("finish at:")
    print(time.ctime(),time.time())
    return y


def normerrorcalc(x,y,testno):
    print("normerrortest:{}".format(testno))
    for i in range(0,NumTrains):
        NormError[testno,i]=np.mean(abs((y-Models[ModelNames[i]].predict(x)[0])/y))




######################################Training################################

#Input ENKI data
XENKI=np.transpose(np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/ENKIparameters400x4.npy"))
YENKI=np.transpose(np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/ENKIOutput400x4.npy"))
#Define parameters
Ntrain=100
M=4
N=5
fasttime=1000
fastspin=100
K=36
J=10
t=np.arange(0.0,fasttime/100,0.01)
Models={} #Dictionary to store models in
NumTrains=5 #Number of training methods
ModelNames=['ENKI','Gaussian','LHS','MCMC','GPMCMC']

####Converged ENKI Training
ENKItrainind=range(1600-Ntrain,1600)
gpmake(XENKI[ENKItrainind],YENKI[ENKItrainind],'ENKI')



####GaussianJitter Training
hsigma=1
Fsigma=10
csigma=1
bsigma=10
sigmas=np.zeros(4)
sigmas[:]=hsigma,Fsigma,csigma,bsigma
cov=np.diag(sigmas**2)
GXtrain=XENKI[ENKItrainind]+np.random.multivariate_normal((0,0,0,0),cov,size=Ntrain)
GYtrain=yfromx(GXtrain,Ntrain)
gpmake(GXtrain,GYtrain,'Gaussian')
np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GJXTrain.npy",GXtrain)
np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GJYTrain.npy",GYtrain)

####LatinHypercube Training
lgdtrain=lhs(M,samples=Ntrain,criterion='maximin')
xlhstrain=np.zeros((Ntrain,M))
xlhstrain[:,0]=lgdtrain[:,0]*6-3
xlhstrain[:,1]=lgdtrain[:,1]*52-16
xlhstrain[:,2]=lgdtrain[:,2]*5+5
xlhstrain[:,3]=lgdtrain[:,3]*52-21
ylhstrain=yfromx(xlhstrain,Ntrain)
gpmake(xlhstrain,ylhstrain,'LHS')
np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/LHSXtrain.npy",xlhstrain)
np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/LHSYtrain.npy",ylhstrain)



####MCMC Training

#Load MCMC training data
XMCMCTrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/MCMCXtrain.npy")
YMCMCTrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/MCMCYtrain.npy")
gpmake(XMCMCTrain,YMCMCTrain,'MCMC')

####GP MCMC Training

#Load GP MCMC training data
XGPMCMCData=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/MCMCGPyL96samples.npy")[54000:,:,:]
GPMCMCind=np.around(np.random.rand(Ntrain)*50000*10).astype(int)
XGPMCMCTrain=XGPMCMCData.reshape(500000,4)[GPMCMCind,:]
YGPMCMCTrain=yfromx(XGPMCMCTrain,Ntrain)
gpmake(XGPMCMCTrain,YGPMCMCTrain,'GPMCMC')
np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GPMCMCXTrain",XGPMCMCTrain)
np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GPMCMCYTrain",YGPMCMCTrain)
####################################Testing####################################
Ntest=20
NumTests=11 #Number of tests
NormError=np.zeros((NumTests,NumTrains))

####Converged ENKI Testing
ENKItestind=range(1600-Ntrain-Ntest,1600-Ntrain)
ENKIxtest=XENKI[ENKItestind]
ENKIytest=YENKI[ENKItestind]

normerrorcalc(ENKIxtest,ENKIytest,0)



####Gaussian Jitter Testing
scales=[0.0,0.25,0.5,0.75,1,2,4]
for i in range(0,7):
    xgaussiantest=XENKI[ENKItestind]+np.random.multivariate_normal((0,0,0,0),scales[i]**2*cov,size=Ntest)
    ygaussiantest=yfromx(xgaussiantest,Ntest)
    normerrorcalc(xgaussiantest,ygaussiantest,i+1)
np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GJxtest.npy",xgaussiantest)
np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GJytest.npy",ygaussiantest)



####LatinHypercube Testing
lgdtest=lhs(4,samples=20,criterion='maximin')
xlhstest=np.zeros((20,4))
xlhstest[:,0]=lgdtest[:,0]*6-3
xlhstest[:,1]=lgdtest[:,1]*52-16
xlhstest[:,2]=lgdtest[:,2]*5+5
xlhstest[:,3]=lgdtest[:,3]*52-21
ylhstest=yfromx(xlhstest,Ntest)
normerrorcalc(xlhstest,ylhstest,8)
np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/LHSxtest.npy",xlhstest)
np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/LHSytest.npy",ylhstest)



####MCMC Testing
MCMCxtest=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/MCMCxtest.npy")
MCMCytest=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/MCMCytest.npy")
normerrorcalc(MCMCxtest,MCMCytest,9)


####GPMCMC Testing
GPMCMCtestind=np.around(np.random.rand(Ntest)*50000*10).astype(int)
XGPMCMCTest=XGPMCMCData.reshape(500000,4)[GPMCMCtestind,:]
YGPMCMCTest=yfromx(XGPMCMCTest,Ntest)
normerrorcalc(XGPMCMCTest,YGPMCMCTest,10)
np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GPMCMCxtest.npy",XGPMCMCTest)
np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GPMCMCytest.npy",YGPMCMCTest)


####Output NormError
np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/NormError.npy",NormError)




























