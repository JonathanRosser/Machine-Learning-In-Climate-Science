import numpy as np
import scipy.stats as st
import seaborn as sns
import GPy
from GPy.models.gp_regression import GPRegression as reg
import matplotlib.pyplot as plt




#Define parameters
sigma=10
longtime=10000
longspin=1000
chains=20
z_ENKI=np.zeros(1)
z_ENKI[0]=2
ITER=1000 #Number of iterations
M=1
N=1
ztrue=np.zeros((M+N))
ztrue[:M]=3
cov_par=100

#Define useful functions
def q(z):
    return np.random.multivariate_normal(z,(sigma**2)*np.identity(1))

def prior(z):
    return st.norm.pdf(z,loc=z_ENKI,scale=10)

def phi(y,z):
    phi=1/2*(y-gp.predict(z.reshape((1,1)))[0])**2*cov_par
    return float(phi)

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
Tsize=100 #Size of training se
X=np.linspace(-100,100,Tsize).reshape((Tsize,1))
Y=X**3+20*np.random.normal(loc=0.0,scale=1.0,size=Tsize).reshape((Tsize,1))
print("Gaussian Process")
gp=reg(X,Y,noise_var=0.1,normalizer=True)
gp.optimize()


#Obtain the long run statistics (true data)
obs=np.zeros((100))
for i in range(0,100):
    obs[i]=ztrue[:M]**3+np.random.normal(loc=0.0,scale=1.0)

ztrue[M:]=np.mean(obs)


#Iterate
print("initial positions")
samples=np.zeros((ITER,chains,M))
#Start the initial Markov chains (In future possibly consider starting these in different places)
for i in range(0,chains):
    samples[0,i,:]=z_ENKI+np.random.normal(loc=0.0,scale=10.0)

#Begin the iteration process

for i in range(1,ITER):
    print(i)
    for j in range(0,chains):
        z=samples[i-1,j,:]
        zstar=q(z)
        samples[i,j,:]=accept(ztrue[M:],zstar,z)




