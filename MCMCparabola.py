import numpy as np
import scipy.stats as st
import seaborn as sns
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.integrate import odeint




#Define parameters
sigma=3
longtime=10000
longspin=1000
chains=4
z_ENKI=np.zeros(1)
z_ENKI[0]=2
ITER=1000 #Number of iterations
M=1
N=1
ztrue=np.zeros((M+N))
ztrue[:M]=3


#Define useful functions
def q(z):
    return np.random.multivariate_normal(z,(sigma**2)*np.identity(1))

def prior(z):
    return st.norm.pdf(z,loc=z_ENKI,scale=10)

def phi(y,z):
    phi=1/2*(y-gp.predict(z.reshape((1,1))))**2
    return phi

def U(y,z):
    U=-np.log(prior(z))+phi(y,z)
    return U

def accept(y,zstar,z):
    Uz=U(y,z)
    Uzstar=U(y,zstar)
    A=np.min((1,np.exp(-Uzstar+Uz)))
    test=np.random.rand()
    print("z=%f , zstar=%f , pz=%f , pzstar=%f , A=%f , test=%f " )%(float(z),float(zstar),float(Uz),float(Uzstar),float(A),float(test)) 
    if A < test:
        print("reject")
        return z
    else:
        print("accept")
        return zstar




#First stage is to train the Gaussian Emulator
#Train on one parameter initially
print("Train GP")
Tsize=10 #Size of training set
X=np.linspace(-100,100,Tsize)
Y=X**2+np.random.normal(loc=0.0,scale=1.0,size=Tsize)
print("Gaussian Process")
gp=gaussian_process.GaussianProcessRegressor(normalize_y=False)
gp.fit(X.reshape((Tsize,1)),Y.reshape((Tsize,1)))

#Obtain the long run statistics (true data)
obs=np.zeros((100))
for i in range(0,100):
    obs[i]=ztrue[:M]**2+np.random.normal(loc=0.0,scale=1.0)

ztrue[M:]=np.mean(obs)


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
        samples[i,j,:]=accept(ztrue[M:],zstar,z)




