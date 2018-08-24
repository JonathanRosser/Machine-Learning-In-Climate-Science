import GPy
import numpy as np
from GPy.models.gp_regression import GPRegression as reg
import matplotlib.pyplot as plt
import scipy.stats as st
from scipy.integrate import odeint

#Define functions which will be useful

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

def prior(z):
    return st.norm.pdf(z[0],loc=0,scale=1)*st.norm.pdf(z[1],loc=10.0,scale=10.0)*st.lognorm.pdf(z[2],0.1,scale=np.exp(2))*st.norm.pdf(z[3],loc=5.0,scale=10.0)
    #return 1

def function(x):
    x0=np.random.rand(K*(J+1))
    y=output(odeint(Lorenz96,x0,t,tuple(x)),fasttime,fastspin)
    return y



def q(z):
    #This is the function which determines how to move around the space, it needs to be symmetric currently because this is using the Metropolis algorithm
    return np.random.multivariate_normal(z,(step_sigma**2)*np.identity(M))



def phi(y,z):
    phi=1.0/2.0*np.sum((y-function(z))**2)
    return phi

def U(y,z):
    U=-np.log(prior(z))+phi(y,z)
    return U

def accept(y,zstar,z,Uz):
    Uzstar=U(y,zstar)
    A=np.min((1,np.exp(-Uzstar+Uz)))
    test=np.random.rand()
    #print("z=%f , zstar=%f , pz=%f , pzstar=%f , A=%f , test=%f " )%(float(z),float(zstar),float(Uz),float(Uzstar),float(A),float(test)) 
    if A < test:
        #print("reject")
        return z,0,Uz
    else:
        #print("accept")
        return zstar,1,Uzstar




M=4
N=5
K=36
J=10
fasttime=1000
fastspin=100
t=np.arange(0.0,fasttime/100,0.01)
XENKI=np.transpose(np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/ENKIparameters400x4.npy"))
YENKI=np.transpose(np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/ENKIOutput400x4.npy"))
ztrue=np.zeros((M+N))
ztrue=np.load("/home/jprosser/Documents/WorkPlacements/Caltech/Data/ztrue.npy")
chains=20
ITER=20

step_sigma=0.8
burnin=0
a=np.zeros((ITER+burnin)*chains)
Ntest=20



for k in range(0,10):
    #Iterate
    print("initial positions")
    samples=np.zeros((ITER+burnin,chains,M))
    ENKItestind=np.empty((Ntest))
    j=-1
    for i in range(0,Ntest):
        while j<i:
            r=np.random.randint(1200,1600)
            if np.sum(ENKItestind==r)==0:
                ENKItestind[j]=r
                print(r)
                j+=1
    ENKItestind=ENKItestind.astype(int)




    #Start the initial Markov chains (In future possibly consider starting these in different places)
    for i in range(0,chains):
        samples[0,i,:]=XENKI[ENKItestind[i],:]

    #Begin the iteration process
    Uvalues=np.zeros((chains))
    for i in range(0,chains):
        Uvalues[i]=U(ztrue[M:],samples[0,i,:])


    for i in range(1,ITER+burnin):
        print(i)
        for j in range(0,chains):
            z=samples[i-1,j,:]
            zstar=q(z)
            samples[i,j,:],a[4*i+j],Uvalues[j]=accept(ztrue[M:],zstar,z,Uvalues[j])

    Xtest=samples[ITER+burnin-1,:,:]





    Ytest=np.zeros((Ntest,N))

    #Use the Lorenz model to calculate the y values
    t=np.arange(0.0,fasttime/100,0.01)
    for i in range(0,Ntest):
        print(i)
        x0=np.random.rand(K*(J+1))
        Ytest[i,:]=output(odeint(Lorenz96,x0,t,tuple(Xtest[i,:])),fasttime,fastspin)


    np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/MCMCXTest{}.npy".format(k),Xtest)
    np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/MCMCYTest{}.npy".format(k),Ytest)





















