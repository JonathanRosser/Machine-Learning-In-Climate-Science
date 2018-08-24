import GPy
import numpy as np
from GPy.models.gp_regression import GPRegression as reg
#import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.integrate import odeint
from pyDOE import *


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
    #return st.norm.pdf(z[0],loc=0,scale=1)*st.norm.pdf(z[1],loc=10.0,scale=10.0)*st.lognorm.pdf(z[2],0.1,scale=np.exp(2))*st.norm.pdf(z[3],loc=5.0,scale=10.0)
    return 1


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
    for j in range(0,Num):
        print(j)
        x0=np.random.rand(K*(J+1))
        y[j,:]=output(odeint(Lorenz96,x0,t,tuple(x[j,:])),fasttime,fastspin)
    return y
    
def normerrorcalc(x,y,testno):
    print("normerrortest:{}".format(testno))
    for i in range(0,NumTrains):
        NormError[testno,i]=np.mean(abs((y-Models[ModelNames[i]].predict(x))/y))


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
        print(i)
        for j in range(0,chains):
            z=samples[i-1,j,:]
            zstar=q(z,step_sigma)
            samples[i,j,:],a[4*i+j],Uvalues[j]=accept(ztrue[M:],zstar,z,Uvalues[j],name)
        if i%10==0 and i<=burnin:
            meana=np.mean(a[4*i+3-10:4*i+3])
            if meana<ideal_accept-accept_bounds or meana>ideal_accept+accept_bounds:
                step_sigma=step_sigma*np.exp(meana-ideal_accept)
                print(meana,step_sigma)
    return samples



######################################Training################################

#Input ENKI data
XENKI=np.transpose(np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/ENKIparameters400x4.npy"))
YENKI=np.transpose(np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/ENKIOutput400x4.npy"))
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
samples=np.zeros((NumTrains,ITER+burnin,chains,M))





####Converged ENKI Training
ENKItrainind=range(1600-Ntrain,1600)
gpmake(XENKI[ENKItrainind],YENKI[ENKItrainind],'ENKI')
samples[0,:,:]=MCMC(ModelNames[0],chains,ITER,burnin)
np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GPCONVENKISAMPLESUniformPrior.npy",samples[0,:,:])



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
samples[1,:,:]=MCMC(ModelNames[1],chains,ITER,burnin)
np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GaussianGPSAMPLESUniformPrior.npy",samples[1,:,:])



####LatinHypercube Training
lgdtrain=lhs(M,samples=Ntrain,criterion='maximin')
xlhstrain=np.zeros((Ntrain,M))
xlhstrain[:,0]=lgdtrain[:,0]*6-3
xlhstrain[:,1]=lgdtrain[:,1]*52-16
xlhstrain[:,2]=lgdtrain[:,2]*5+5
xlhstrain[:,3]=lgdtrain[:,3]*52-21
ylhstrain=yfromx(xlhstrain,Ntrain)
gpmake(xlhstrain,ylhstrain,'LHS')
samples[2,:,:]=MCMC(ModelNames[2],chains,ITER,burnin)
np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GPLHSSAMPLESUniformPrior.npy",samples[2,:,:])




####MCMC Training

#Load MCMC training data
XMCMCTrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/MCMCXtrain.npy")
YMCMCTrain=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/MCMCYtrain.npy")
gpmake(XMCMCTrain,YMCMCTrain,'MCMC')
samples[3,:,:]=MCMC(ModelNames[3],chains,ITER,burnin)
np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GPMCMCSAMPLESUniformPrior.npy",samples[3,:,:])



####GP MCMC Training

#Load GP MCMC training data
XGPMCMCData=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/MCMCGPyL96samples.npy")[54000:,:,:]
GPMCMCind=np.around(np.random.rand(Ntrain)*50000*10).astype(int)
XGPMCMCTrain=XGPMCMCData.reshape(500000,4)[GPMCMCind,:]
YGPMCMCTrain=yfromx(XGPMCMCTrain,Ntrain)
gpmake(XGPMCMCTrain,YGPMCMCTrain,'GPMCMC')
samples[4,:,:]=MCMC(ModelNames[4],chains,ITER,burnin)
np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GPGPMCMCTRAINSAMPLESUniformPrior.npy",samples[4,:,:])



####Full ENKI Training

XFULLENKI=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/ENKIparameters400x4.npy")
YFULLENKI=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/ENKIOutput400x4.npy")
gpmake(XFULLENKI,YFULLENKI,'FULLENKI')
samples[5,:,:]=MCMC(ModelNames[5],chains,ITER,burnin)
np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GPMCMCFULLENKISAMPLESUniformPrior.npy",samples[5,:,:])



##############################MCMC#####################################
#Define parameters
z_ENKI=np.zeros(4)
z_ENKI[0]=1
z_ENKI[1]=10
z_ENKI[2]=10
z_ENKI[3]=10
chains=10
ITER=10000
burnin=2000
#samples=np.zeros((NumTrains,ITER+burnin,chains,M))
#for i in range(0,NumTrains):	
#	print(i)
#    samples[i,:,:]=MCMC(ModelNames[i],chains,ITER,burnin)

np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/Combined/GPMCMCsamplesUniformPrior.npy",samples)




