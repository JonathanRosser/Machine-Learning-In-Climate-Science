import numpy as np
import scipy.stats as st
import seaborn as sns
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.integrate import odeint




#Define parameters
sigma=100
longtime=10000
longspin=1000
chains=4
z_ENKI=np.zeros(1)
z_ENKI[0]=11
ITER=1000 #Number of iterations
M=1
N=1
K=36 #Number of slow variables
J=10 #Number of fast variables per slow variable
ztrue=np.zeros((M+N))
ztrue[:M]=10


#Define useful functions
def q(z):
    return np.random.multivariate_normal(z,(sigma**2)*np.identity(1))
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
    y=np.zeros(N)
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


def accept(zstar,z):
    A=np.min((1,st.norm.pdf(ztrue[M],loc=gp.predict(zstar.reshape((1,1)),return_std=True)[0],scale=gp.predict(zstar.reshape((1,1)),return_std=True)[1])/st.norm.pdf(ztrue[M],loc=gp.predict(z.reshape((1,1)),return_std=True)[0],scale=gp.predict(z.reshape((1,1)),return_std=True)[1])))
    test=np.random.rand()
    if A < test:
        print("reject")
        return z
    else:
        print("accept")
        return zstar




#First stage is to train the Gaussian Emulator
#Train on one parameter initially
print("Train GP")
X=np.load("/home/jonathan/Documents/WorkPlacements/Caltech/Data/parametersconstF.npy")[:,1]
Y=np.load("/home/jonathan/Documents/WorkPlacements/Caltech/Data/OutputconstF.npy")[:,0]
print("Gaussian Process")
gp=gaussian_process.GaussianProcessRegressor(normalize_y=True)
gp.fit(X.reshape((100,1)),Y.reshape((100,1)))

#Obtain the long run statistics (true data)
ztrue=np.load("/home/jonathan/Documents/WorkPlacements/Caltech/Data/ztrue.npy")[1:5:3]


#Iterate
print("initial positions")
samples=np.zeros((ITER,chains,M))
#Start the initial Markov chains (In future possibly consider starting these in different places)
for i in range(0,chains):
    samples[0,i,:]=z_ENKI+np.random.normal(loc=0.0,scale=1.0)

#Begin the iteration process

for i in range(1,ITER):
    print(i)
    for j in range(0,chains):
        z=samples[i-1,j,:]
        zstar=q(z)
        samples[i,j,:]=accept(zstar,z)




