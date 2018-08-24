import GPy
import numpy as np
from GPy.models.gp_regression import GPRegression as reg
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.integrate import odeint
from pyDOE import *
import time
import scipy.io



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
        x0=np.random.rand(K*(J+1))
        y[j,:]=output(odeint(Lorenz96,x0,t,tuple(x[j,:])),fasttime,fastspin)
    print("finish at:")
    print(time.ctime(),time.time())
    return y


def normerrorcalc(x,y,testno,trainsamp,testsamp):
    print("normerrortest:{}".format(testno),type(x),type(y),type(testno),type(trainsamp),type(testsamp))
    for i in range(0,NumTrains):
        NormError[trainsamp,testsamp,testno,i]=np.mean(abs((y-Models[ModelNames[i]].predict(x)[0])/y))

NumTrains=5 #Number of training methods
NumSamples=10
Ntest=20
NumTests=13 #Number of tests
NormError=np.zeros((NumSamples,NumSamples,NumTests,NumTrains))

for SAMPLE in range(0,NumSamples):

	######################################Training################################
	print("SAMPLE = {}".format(SAMPLE))
	#Input ENKI data
	if SAMPLE < NumSamples/4.0:
		print("ENKI 0")
		XENKI=np.transpose(np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/ENKIparameters400x4.npy"))
		YENKI=np.transpose(np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/ENKIOutput400x4.npy"))
	elif SAMPLE< NumSamples/2.0:
		print("ENKI 1")
		XENKI=np.transpose(np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/ENKIparameters400x4sample2.npy"))
                YENKI=np.transpose(np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/ENKIOutput400x4sample2.npy"))
	elif SAMPLE < NumSamples*3.0/4.0:
		print("ENKI 2")
                XENKI=np.transpose(np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/ENKIparameters400x4sample3.npy"))
                YENKI=np.transpose(np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/ENKIOutput400x4sample3.npy"))
	else:
		print("ENKI 3")
                XENKI=np.transpose(np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/ENKIparameters400x4sample4.npy"))
                YENKI=np.transpose(np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/ENKIOutput400x4sample4.npy"))


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
	ModelNames=['ENKI','Gaussian','LHS','MCMC','GPMCMC']

	####Converged ENKI Training
	gpmake(XENKI,YENKI,'ENKI')



	####GaussianJitter Training
	GXtrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GJXTrain{}.npy".format(SAMPLE))
	GYtrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GJYTrain{}.npy".format(SAMPLE))
	gpmake(np.append(GXtrain,XENKI,axis=0),np.append(GYtrain,YENKI,axis=0),'Gaussian')




	####LatinHypercube Training
	xlhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/LHSXTrain{}.npy".format(SAMPLE))
	ylhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/LHSYTrain{}.npy".format(SAMPLE))
	gpmake(np.append(xlhstrain,XENKI,axis=0),np.append(ylhstrain,YENKI,axis=0),'LHS')



	####MCMC Training

	#Load MCMC training data
	XMCMCTrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/MCMCXTrain{}.npy".format(SAMPLE))
	YMCMCTrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/MCMCYTrain{}.npy".format(SAMPLE))
	gpmake(np.append(XMCMCTrain,XENKI,axis=0),np.append(YMCMCTrain,YENKI,axis=0),'MCMC')

	####GP MCMC Training

	#Load GP MCMC training data
	GPSAMPLES= np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GPCONVENKISAMPLES.npy")[2000:,:,:].reshape((100000,4))
	#Create training data
	nselect=np.empty((Ntrain))
	j=-1
	for i in range(0,Ntrain):
	    while j<i:
		r=np.random.randint(0,len(GPSAMPLES))
		if np.sum(nselect==r)==0:
		    nselect[j]=r
		    j+=1
	nselect=nselect.astype(int)
	GPMCMCxtrain=GPSAMPLES[nselect].reshape((Ntrain,M))
	GPMCMCytrain=yfromx(GPMCMCxtrain,Ntrain)
	np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GPMCMCXTrain{}.npy".format(SAMPLE),GPMCMCxtrain)
	np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GPMCMCYTrain{}.npy".format(SAMPLE),GPMCMCytrain)
	gpmake(np.append(GPMCMCxtrain,XENKI,axis=0),np.append(GPMCMCytrain,YENKI,axis=0),'GPMCMC')



	for TESTSAMPLE in range(0,NumSamples):
		####################################Testing####################################
		print("TESTSAMPLE = {}".format(TESTSAMPLE))

		####Converged ENKI Testing
		#Create training data
		ntestselect=np.empty((Ntest))
		j=-1
		for i in range(0,Ntest):
		    while j<i:
			r=np.random.randint(1200,1600)
			if np.sum(ntestselect==r)==0:
			    ntestselect[j]=r
			    print(r)
			    j+=1
		ntestselect=ntestselect.astype(int)

		ENKIxtest=XENKI[ntestselect]
		ENKIytest=YENKI[ntestselect]

		normerrorcalc(ENKIxtest,ENKIytest,0,SAMPLE,TESTSAMPLE)



		####Gaussian Jitter Testing
		xgaussiantest=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GJXTest{}.npy".format(TESTSAMPLE))
		ygaussiantest=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GJYTest{}.npy".format(TESTSAMPLE))


		for i in range(0,7):
		    normerrorcalc(xgaussiantest[i,:,:],ygaussiantest[i,:,:],i+1,SAMPLE,TESTSAMPLE)




		####LatinHypercube Testing
		xlhstest=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/LHSXTest{}.npy".format(TESTSAMPLE))
		ylhstest=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/LHSYTest{}.npy".format(TESTSAMPLE))
		normerrorcalc(xlhstest,ylhstest,8,SAMPLE,TESTSAMPLE)



		####MCMC Testing
		MCMCxtest=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/MCMCXTest{}.npy".format(TESTSAMPLE))
		MCMCytest=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/MCMCYTest{}.npy".format(TESTSAMPLE))
		normerrorcalc(MCMCxtest,MCMCytest,9,SAMPLE,TESTSAMPLE)


		####GPMCMC Testing
		XGPMCMCTest=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GPMCMCxtest.npy")
		YGPMCMCTest=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GPMCMCytest.npy")
		normerrorcalc(XGPMCMCTest,YGPMCMCTest,10,SAMPLE,TESTSAMPLE)


		####ShiWei Testing
		xswtest=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/xswtest{}.npy".format(TESTSAMPLE))
		yswtest=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/yswtest{}.npy".format(TESTSAMPLE))
		normerrorcalc(xswtest,yswtest,11,SAMPLE,TESTSAMPLE)



		####New GP test outputs
		#Create training data
		GPMCMCxtest=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GPMCMCXTest{}.npy".format(TESTSAMPLE))
		GPMCMCytest=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GPMCMCYTest{}.npy".format(TESTSAMPLE))

		normerrorcalc(GPMCMCxtest,GPMCMCytest,12,SAMPLE,TESTSAMPLE)
	np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/NormError1700samp{}.npy".format(SAMPLE),NormError)
####Output NormError
np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/NormError1700.npy",NormError)




























