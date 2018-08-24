import GPy
import numpy as np
from GPy.models.gp_regression import GPRegression as reg
import matplotlib.pyplot as plt
import scipy.stats as st


#Define input data (observations)
X=np.load("/home/jonathan/Documents/WorkPlacements/Caltech/Data/ENKIparametersconstF.npy")[1,:]
Y=np.load("/home/jonathan/Documents/WorkPlacements/Caltech/Data/ENKIOutputconstF.npy")[0,:]  

#define seed:
seed=3
np.random.seed(seed=seed)


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
print("test data created")

#Create gp
L=np.zeros((162,5))
N=0
for M in [1]:
    print("defining kernels")
    kerns = [GPy.kern.RBF(M), GPy.kern.Exponential(M), GPy.kern.Matern32(M), GPy.kern.Matern52(M), GPy.kern.Brownian(M), GPy.kern.Bias(M), GPy.kern.Linear(M), GPy.kern.PeriodicExponential(M), GPy.kern.White(M)]
    for p in range(0,9):
        print("defining specific kernel")
        k=kerns[p]
        for i in [0.0,0.1,0.5,1,10,100]:
            for V in [True,False,None]:
                gp=reg(xtrain,ytrain,k,noise_var=i,normalizer=V)
                gp.optimize()
                #print(M,k,i,V)
                #print(gp)
                ymodel=gp.predict(xtest)
                #meandiff=np.abs(np.mean(ytest-ymodel[0]))
                meansqdiff=np.mean((ytest-ymodel[0])**2)
                fig=gp.plot()
                GPy.plotting.show(fig)
                plt.scatter(xtest,ytest)
                plt.savefig("/home/jonathan/Documents/Temporary/GPytestplots/rbfplot{}.png".format(N))
                plt.close()
                #meanstddiff=np.abs(np.mean((ytest-ymodel[0])/ymodel[1]))
                #meanstdsqdiff=np.mean(((ytest-ymodel[0])/ymodel[1])**2)
                #pdf=np.prod(st.norm.pdf(ytest,loc=ymodel[0],scale=ymodel[1]))
                #gpcont=()
                #B=0
                #while True:
                #    try:
                #        gpcont=np.append(gpcont,gp[B])
                #        B+=1
                #    except IndexError:
                #        break
                #m=len(gpcont)
                #chi=np.abs(np.sum(((ytest-ymodel[0])**2)/ymodel[1])/(len(ntest)-m)-1)
                #L[N,:]=N,gp.log_likelihood(),p,i,V,meandiff,meansqdiff,meanstddiff,meanstdsqdiff,pdf,chi
                L[N,:]=N,p,i,V,meansqdiff
                print(L[N,:],k, np.mean(xtrain),np.mean(ytrain),np.mean(xtest),np.mean(ytest),np.mean(ymodel))
                N+=1
                del gp

for i in range(0,len(L)):
    if L[i,0]==np.max(L[:,0]):
        print(L[i,:])

Lindex=np.zeros((5,np.shape(L)[0]))
Lfinal=np.zeros((5,np.shape(L)[0],np.shape(L)[1]))
for i in range(0,5):
    Lindex[i,:]=np.argsort(L[:,i],axis=0)
    Lfinal[i,:,:]=L[Lindex[i,:].astype(int)]

def gptest(p,i,v):
    if v==0:
        V=False
    elif v==1:
        V=True
    else: 
        V=None
    gp=reg(xtrain,ytrain,kerns[p],noise_var=i,normalizer=V)
    gp.optimize()
    ymodel=gp.predict(xtest)
    meansqdiff=np.mean((ytest-ymodel[0])**2)
    print(meansqdiff)
    fig=gp.plot()
    GPy.plotting.show(fig)
    plt.scatter(xtest,ytest)
    plt.show()

#plot gp

#fig=gp.plot()

#GPy.plotting.show(fig)
#plt.show()

