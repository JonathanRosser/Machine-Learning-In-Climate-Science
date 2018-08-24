import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.integrate import odeint

#Create the Gaussian Process
X=np.load("/home/jonathan/Documents/WorkPlacements/Caltech/Data/parameters.npy")[:1000,:]
Y=np.load("/home/jonathan/Documents/WorkPlacements/Caltech/Data/Output.npy")[:1000,:]
print("Gaussian Process")
gp=gaussian_process.GaussianProcessRegressor(normalize_y=True)
gp.fit(X,Y[:,0])



#Generate the data with true parameter values
true=np.zeros((1,4))
true[0,0]=1  #h
true[0,1]=10 #F
true[0,2]=10 #c
true[0,3]=10 #b
size=100



Observed_data=gp.sample_y(true,n_samples=size)

print('Running on PyMC3 v{}'.format(pm.__version__))

basic_model = pm.Model()

with basic_model:

    #Priors for unknown model parameters
    h=pm.Normal('h', mu=0.0, sd=1.0)
    F=pm.Normal('F', mu=10.0, sd=10.0)
    c=np.exp(pm.Normal('c', mu=2.0, sd=0.1))
    b=pm.Normal('b', mu=5.0, sd=10.0)
    print('set priors')
    #X=np.zeros((1,4))
    #X[0,0]=h
    #X[0,1]=F
    #X[0,2]=c
    #X[0,3]=b
    X2=((h,F,c,b),)
    mu=gp.predict(X2)
    sigma=gp.predict(X2,return_std=True)[1]

    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=Observed_data)
    print('sampling stage')
    trace=pm.sample(10000)
    
pm.traceplot(trace)
plt.show()

