import GPy
import numpy as np
from GPy.models.gp_regression import GPRegression as reg
import matplotlib.pyplot as plt
import scipy.stats as st


#Define input data (observations)
X=np.load("/home/jonathan/Documents/WorkPlacements/Caltech/Data/ENKIparametersconstF.npy")[1,:]
Y=np.load("/home/jonathan/Documents/WorkPlacements/Caltech/Data/ENKIOutputconstF.npy")[0,:]  


#Create training data
ntrain=80
nselect=np.empty((ntrain))
j=-1
for i in range(0,ntrain):
    while j<i:
        r=np.random.randint(0,len(X))
        if np.sum(nselect==r)==0:
            nselect[j]=r
            print(r)
            j+=1
nselect=nselect.astype(int)
xtrain=X[nselect].reshape((ntrain,1))
ytrain=Y[nselect].reshape((ntrain,1))

#Create test data
ntest=np.zeros((len(X)-ntrain))
xtest=np.zeros((len(X)-ntrain,1))
ytest=np.zeros((len(X)-ntrain,1))
j2=0
for i in range(0,len(X)):
    if np.sum(nselect==i)==0:
        ntest[j2]=i
        j2+=1
ntest=ntest.astype(int)
xtest=X[ntest].reshape((len(X)-ntrain,1))
ytest=Y[ntest].reshape((len(X)-ntrain,1))


#Create gp
L=np.zeros((162,5))
N=0
for M in [1]:
    kerns = [GPy.kern.RBF(M), GPy.kern.Exponential(M), GPy.kern.Matern32(M), GPy.kern.Matern52(M), GPy.kern.Brownian(M), GPy.kern.Bias(M), GPy.kern.Linear(M), GPy.kern.PeriodicExponential(M), GPy.kern.White(M)]
    for p in range(0,9):
        k=kerns[p]
        for i in [0.0,0.1,0.5,1,10,100]:
            for V in [True,False,None]:
                gp=reg(xtrain,ytrain,k,noise_var=i,normalizer=V)
                gp.optimize()
                #print(M,k,i,V)
                #print(gp)
                ymodel=gp.predict(xtest)
                #meandiff=np.abs(np.mean(ytest-ymodel[0]))
                #meansqdiff=np.mean((ytest-ymodel[0])**2)
                #meanstddiff=np.abs(np.mean((ytest-ymodel[0])/ymodel[1]))
                #meanstdsqdiff=np.mean(((ytest-ymodel[0])/ymodel[1])**2)
                #pdf=np.prod(st.norm.pdf(ytest,loc=ymodel[0],scale=ymodel[1]))
                gpcont=()
                B=0
                while True:
                    try:
                        gpcont=np.append(gpcont,gp[B])
                        B+=1
                    except IndexError:
                        break
                m=len(gpcont)
                chi=np.abs(np.sum(((ytest-ymodel[0])**2)/ymodel[1])/(len(ntest)-m)-1)
                L[N,:]=gp.log_likelihood(),p,i,V,chi
                print(L[N,:])
                N+=1

for i in range(0,len(L)):
    if L[i,0]==np.max(L[:,0]):
        print(L[i,:])

Lindex=np.zeros((5,np.shape(L)[0]))
Lfinal=np.zeros((5,np.shape(L)[0],np.shape(L)[1]))
for i in range(0,5):
    Lindex[i,:]=np.argsort(L[:,i],axis=0)
    Lfinal[i,:,:]=L[Lindex[i,:].astype(int)]



#plot gp

#fig=gp.plot()

#GPy.plotting.show(fig)
#plt.show()

