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
    if name=='ENKI':
	gp=reg(X,Y,kernel,noise_var=noise,normalizer=normalizer)
	gp.optimize()
	Models[name]=gp
    else:
    	gp=reg(X,Y,kernel,noise_var=noise,normalizer=normalizer)
        gp.optimize()
        Models[name+'E500']=gp
	del gp
	gp2=reg(np.append(XENKI,X,axis=0),np.append(YENKI,Y,axis=0),kernel,noise_var=noise,normalizer=normalizer)
	gp2.optimize()
	Models[name+'E2100']=gp2
	del gp2

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
        #NormError[trainsamp,testsamp,testno,i]=np.mean(abs((y-Models[ModelNames[i]].predict(x)[0])/y))
	NormError[trainsamp,testsamp,testno,i]=np.mean((y-Models[ModelNames[i]].predict(x)[0])**2)

NumTrains=21 #Number of training methods
NumSamples=2
Ntest=500
NumTests=10 #Number of tests
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
	ModelNames=['ENKI','Gaussian1E500','Gaussian1E2100','Gaussian0.5E500','Gaussian0.5E2100','Gaussian0.1E500','Gaussian0.1E2100','LHS1E500','LHS1E2100','LHS0.5E500','LHS0.5E2100','LHS0.1E500','LHS0.1E2100','LHSMSE500','LHSMSE2100','LHSM0.1E500','LHSM0.1E2100','MCMCE500','MCMCE2100','GPMCMCE500','GPMCMCE2100']

	####Converged ENKI Training
	gpmake(XENKI,YENKI,'ENKI')



	####GaussianJitter Training
	GXtrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/GJXTrain{}.npy".format(SAMPLE))
	GYtrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/GJYTrain{}.npy".format(SAMPLE))
	gpmake(GXtrain,GYtrain,'Gaussian1')

        GXtrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/GJ0p5XTrain{}.npy".format(SAMPLE))
        GYtrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/GJ0p5YTrain{}.npy".format(SAMPLE))
        gpmake(GXtrain,GYtrain,'Gaussian0.5')

        GXtrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/GJ0p1XTrain{}.npy".format(SAMPLE))
        GYtrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/GJ0p1YTrain{}.npy".format(SAMPLE))
        gpmake(GXtrain,GYtrain,'Gaussian0.1')






	####LatinHypercube Training
	xlhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/LHSXTrain{}.npy".format(SAMPLE))
	ylhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/LHSYTrain{}.npy".format(SAMPLE))
	gpmake(xlhstrain,ylhstrain,'LHS1')

        xlhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/LHS0p5XTrain{}.npy".format(SAMPLE))
        ylhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/LHS0p5YTrain{}.npy".format(SAMPLE))
        gpmake(xlhstrain,ylhstrain,'LHS0.5')

        xlhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/LHS0p1XTrain{}.npy".format(SAMPLE))
        ylhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/LHS0p1YTrain{}.npy".format(SAMPLE))
        gpmake(xlhstrain,ylhstrain,'LHS0.1')

        xlhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/LHSENKIMSXTrain{}.npy".format(SAMPLE))
        ylhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/LHSENKIMSYTrain{}.npy".format(SAMPLE))
        gpmake(xlhstrain,ylhstrain,'LHSMS')

        xlhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/LHSENKIM0p1XTrain{}.npy".format(SAMPLE))
        ylhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/LHSENKIM0p1YTrain{}.npy".format(SAMPLE))
        gpmake(xlhstrain,ylhstrain,'LHSM0.1')



	####MCMC Training

	#Load MCMC training data
	XMCMCTrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/MCMCXTrain{}.npy".format(SAMPLE))
	YMCMCTrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/MCMCYTrain{}.npy".format(SAMPLE))
	gpmake(XMCMCTrain,YMCMCTrain,'MCMC')

	####GP MCMC Training

	#Load GP MCMC training data
        XGPMCMCTrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/GPMCMCXTrain{}.npy".format(SAMPLE))
        YGPMCMCTrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/GPMCMCYTrain{}.npy".format(SAMPLE))
	gpmake(XGPMCMCTrain,YGPMCMCTrain,'GPMCMC')



	for TESTSAMPLE in range(0,NumSamples):
		####################################Testing####################################
		print("TESTSAMPLE = {}".format(TESTSAMPLE))


		####GaussianJitter Testing
		GXtrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/GJXTrain{}.npy".format(TESTSAMPLE))
		GYtrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/GJYTrain{}.npy".format(TESTSAMPLE))
		normerrorcalc(GXtrain,GYtrain,0,SAMPLE,TESTSAMPLE)

		GXtrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/GJ0p5XTrain{}.npy".format(TESTSAMPLE))
		GYtrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/GJ0p5YTrain{}.npy".format(TESTSAMPLE))
		normerrorcalc(GXtrain,GYtrain,1,SAMPLE,TESTSAMPLE)

		GXtrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/GJ0p1XTrain{}.npy".format(TESTSAMPLE))
		GYtrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/GJ0p1YTrain{}.npy".format(TESTSAMPLE))
		normerrorcalc(GXtrain,GYtrain,2,SAMPLE,TESTSAMPLE)

		####LatinHypercube Testing
		xlhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/LHSXTrain{}.npy".format(TESTSAMPLE))
		ylhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/LHSYTrain{}.npy".format(TESTSAMPLE))
		normerrorcalc(xlhstrain,ylhstrain,3,SAMPLE,TESTSAMPLE)

		xlhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/LHS0p5XTrain{}.npy".format(TESTSAMPLE))
		ylhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/LHS0p5YTrain{}.npy".format(TESTSAMPLE))
		normerrorcalc(xlhstrain,ylhstrain,4,SAMPLE,TESTSAMPLE)

		xlhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/LHS0p1XTrain{}.npy".format(TESTSAMPLE))
		ylhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/LHS0p1YTrain{}.npy".format(TESTSAMPLE))
		normerrorcalc(xlhstrain,ylhstrain,5,SAMPLE,TESTSAMPLE)

		xlhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/LHSENKIMSXTrain{}.npy".format(TESTSAMPLE))
		ylhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/LHSENKIMSYTrain{}.npy".format(TESTSAMPLE))
		normerrorcalc(xlhstrain,ylhstrain,6,SAMPLE,TESTSAMPLE)

		xlhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/LHSENKIM0p1XTrain{}.npy".format(TESTSAMPLE))
		ylhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/LHSENKIM0p1YTrain{}.npy".format(TESTSAMPLE))
		normerrorcalc(xlhstrain,ylhstrain,7,SAMPLE,TESTSAMPLE)

		####MCMC Training

		#Load MCMC training data
		XMCMCTrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/MCMCXTrain{}.npy".format(TESTSAMPLE))
		YMCMCTrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/MCMCYTrain{}.npy".format(TESTSAMPLE))
		normerrorcalc(XMCMCTrain,YMCMCTrain,8,SAMPLE,TESTSAMPLE)

		####GP MCMC Training

		#Load GP MCMC training data
		XGPMCMCTrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/GPMCMCXTrain{}.npy".format(TESTSAMPLE))
		YGPMCMCTrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/GPMCMCTrain/GPMCMCYTrain{}.npy".format(TESTSAMPLE))
		normerrorcalc(XGPMCMCTrain,YGPMCMCTrain,9,SAMPLE,TESTSAMPLE)




	np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/FinalGPmeansqdelsamp{}.npy".format(SAMPLE),NormError)
####Output NormError
np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/FinalGPmeansqdelsamp.npy",NormError)




























