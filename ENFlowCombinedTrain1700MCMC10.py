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

def function(x,name):
    #print(x,np.shape(x))
    #Create the function which is being sampled from
    y=Models[name].predict(x.reshape(1,4))[0]
    return y

def q(z,step_sigma):
    #This is the function which determines how to move around the space, it needs to be symmetric currently because this is using the Metropolis algorithm
    return np.random.multivariate_normal(z,(step_sigma**2)*np.identity(M))


def prior(z):
    #Define a chosen prior
    return st.norm.pdf(z[0],loc=0,scale=1)*st.norm.pdf(z[1],loc=10.0,scale=10.0)*st.lognorm.pdf(z[2],0.1,scale=np.exp(2))*st.norm.pdf(z[3],loc=5.0,scale=10.0)
    #return 1


def phi(y,z,name):
    phi=1.0/2.0*np.sum((y-function(z,name))**2)
    return phi

def U(y,z,name):
    #print(y,z,name)
    U=-np.log(prior(z))+phi(y,z,name)
    return U

def accept(y,zstar,z,Uz,name):
    Uzstar=U(y,zstar,name)
    A=np.min((1,np.exp(-Uzstar+Uz)))
    test=np.random.rand()
    #print("z=%f , zstar=%f , pz=%f , pzstar=%f , A=%f , test=%f " )%(float(z),float(zstar),float(Uz),float(Uzstar),float(A),float(test)) 
    if A < test:
        #print("reject")
        return z,0,Uz
    else:
        #print("accept")
        return zstar,1,Uzstar



def MCMC(name,chains,ITER,burnin):
    a=np.zeros((ITER+burnin)*chains)
    ideal_accept=0.2
    accept_bounds=0.05
    step_sigma=0.7  #initial step size
    M=4 #Dimensions of sample space of the function
    N=5 #Dimensions of output space of the function
    print("initial positions")
    samples=np.zeros((ITER+burnin,chains,M))
    #Start the initial Markov chains (In future possibly consider starting these in different places)
    for i in range(0,chains):
        samples[0,i,:]=z_ENKI+np.random.multivariate_normal((0,0,0,0),0.1*np.identity(4))

    #Begin the iteration process
    Uvalues=np.zeros((chains))
    for i in range(0,chains):
        Uvalues[i]=U(ztrue[M:],samples[0,i,:],name)

    for i in range(1,ITER+burnin):
        #print(i)
        for j in range(0,chains):
            z=samples[i-1,j,:]
            zstar=q(z,step_sigma)
            samples[i,j,:],a[4*i+j],Uvalues[j]=accept(ztrue[M:],zstar,z,Uvalues[j],name)
        if i%10==0 and i<=burnin:
            meana=np.mean(a[4*i+3-10:4*i+3])
            if meana<ideal_accept-accept_bounds or meana>ideal_accept+accept_bounds:
                step_sigma=step_sigma*np.exp(meana-ideal_accept)
                #print(meana,step_sigma)
    return samples

#Define parameters
ztrue=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/ztrue.npy")
Ntrain=100
M=4
N=5
fasttime=1000
fastspin=100
K=36
J=10
t=np.arange(0.0,fasttime/100,0.01)
Models={} #Dictionary to store models in
NumTrains=6 #Number of training methods
ModelNames=['ENKI','Gaussian','LHS','MCMC','GPMCMC','FULLENKI']
z_ENKI=np.zeros(4)
z_ENKI[0]=1
z_ENKI[1]=10
z_ENKI[2]=10
z_ENKI[3]=10
chains=10
ITER=100000
burnin=20000
NumSamples=10
samples=np.zeros((NumSamples,NumTrains,ITER+burnin,chains,M))




for SAMPLE in range(0,NumSamples):

	######################################Training################################
	print("SAMPLE = {}".format(SAMPLE))
	#Input ENKI data

	if SAMPLE < NumSamples/4.0:
		ENKINUM=0
	elif SAMPLE< NumSamples/2.0:
		ENKINUM=1
	elif SAMPLE< NumSamples*3.0/4.0:
		ENKINUM=2
	else:
		ENKINUM=3
	XENKI=np.transpose(np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/ENFlow/ENFlowparameters400x4sample{}.npy".format(ENKINUM)))
	YENKI=np.transpose(np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/ENFlow/ENFlowOutput400x4sample{}.npy".format(ENKINUM)))


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
	samples[SAMPLE,0,:,:]=MCMC(ModelNames[0],chains,ITER,burnin)



	####GaussianJitter Training
	GXtrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GJXTrain{}.npy".format(SAMPLE))
	GYtrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GJYTrain{}.npy".format(SAMPLE))
	gpmake(np.append(GXtrain,XENKI,axis=0),np.append(GYtrain,YENKI,axis=0),'Gaussian')
	samples[SAMPLE,1,:,:]=MCMC(ModelNames[1],chains,ITER,burnin)




	####LatinHypercube Training
	xlhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/LHSXTrain{}.npy".format(SAMPLE))
	ylhstrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/LHSYTrain{}.npy".format(SAMPLE))
	gpmake(np.append(xlhstrain,XENKI,axis=0),np.append(ylhstrain,YENKI,axis=0),'LHS')
	samples[SAMPLE,2,:,:]=MCMC(ModelNames[2],chains,ITER,burnin)



	####MCMC Training

	#Load MCMC training data
	XMCMCTrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/MCMCXTrain{}.npy".format(SAMPLE))
	YMCMCTrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/MCMCYTrain{}.npy".format(SAMPLE))
	gpmake(np.append(XMCMCTrain,XENKI,axis=0),np.append(YMCMCTrain,YENKI,axis=0),'MCMC')
	samples[SAMPLE,3,:,:]=MCMC(ModelNames[3],chains,ITER,burnin)

	####GP MCMC Training

	#Load GP MCMC training data
	GPMCMCxtrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GPMCMCXTrain{}.npy".format(SAMPLE))
	GPMCMCytrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GPMCMCYTrain{}.npy".format(SAMPLE))
	gpmake(np.append(GPMCMCxtrain,XENKI,axis=0),np.append(GPMCMCytrain,YENKI,axis=0),'GPMCMC')
	samples[SAMPLE,4,:,:]=MCMC(ModelNames[4],chains,ITER,burnin)
	
	np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GPMCMCsamples1700Uniform.npy",samples)











