import numpy as np
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.integrate import odeint
import matplotlib.pyplot as plt
#Take samples with some noise from a known distribution
#Use parabolas
#1D

ITER=100

z=np.zeros((100,3))

z[:,:2]=np.random.uniform(-10,10,(100,2))

z[:,2]=z[:,0]**2+z[:,1]**2+np.random.normal(loc=0.0,scale=1.0)


gp=gaussian_process.GaussianProcessRegressor()
gp.fit(z[:,:2].reshape((100,2)),z[:,2].reshape((100,1)))
x_pred=np.random.uniform(-11,11,(100,2))
output=gp.sample_y(x_pred)

