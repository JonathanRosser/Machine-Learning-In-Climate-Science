import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm

#test for a linear model and check that we can return the values

#Generate the data with true parameter values
alpha,sigma=1,1
beta=[1,2.5]

size=100

X1=np.random.randn(size)
X2=np.random.randn(size)*0.2

banana=alpha+beta[0]*X1+beta[1]*X2+np.random.randn(size)*sigma

print('Running on PyMC3 v{}'.format(pm.__version__))

basic_model = pm.Model()

with basic_model:

    #Priors for unknown model parameters
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=10, shape=2)
    sigma = pm.HalfNormal('sigma',sd=1)

    mu=alpha+beta[0]*X1 + beta[1]*X2

    Y_obs = pm.Normal('Y_obs', mu=mu, sd=sigma, observed=banana)

    trace=pm.sample(10000)

pm.traceplot(trace)

