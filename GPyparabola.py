import GPy
import numpy as np
from GPy.models.gp_regression import GPRegression as reg
import matplotlib.pyplot as plt


#Define input data (observations)
xmax=10
xmin=-10
N=50
X=np.linspace(xmin,xmax,N).reshape((N,1))
Y=X**2+np.random.normal(size=N).reshape((N,1))



#Create gp

k=GPy.kern.Brownian(input_dim=1)

gp=reg(X,Y,k,noise_var=1,normalizer=True)

#plot gp

fig=gp.plot()

GPy.plotting.show(fig)
plt.show()

