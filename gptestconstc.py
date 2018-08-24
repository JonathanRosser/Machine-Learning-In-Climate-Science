import numpy as np
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from scipy.integrate import odeint


X=np.load("parameters.npy")[:1000,:]
Y=np.load("Output.npy")[:1000,:]
print("Gaussian Process")
gp=gaussian_process.GaussianProcessRegressor(normalize_y=True)
gp.fit(X,Y)


ITER=100 #Number of times to run the Lorenz model
N=5
def output(x,time,spin):
    y=np.zeros(N)
    y[0]=np.mean(x[spin:,:K]) #Mean of the slow variables
    y[1]=np.mean(x[spin:,K:]) #Mean of the fast variables
    y[2]=np.mean(np.square(x[spin:,:K])) #Mean of the squared slow variables
    y[3]=np.mean(np.square(x[spin:,K:])) #Mean of the squared fast variables
    y[4]=XY2(x,time,spin) #Mean of XY 
    return y

def XY2(x,time,spin):
    y=np.zeros((time-spin,K))
    for i in range(0,K):
        y[:,i]=np.mean(x[spin:,K+i*J:K+(i+1)*J],axis=1)
    XY=np.mean(np.multiply(x[spin:,:K],y))
    return XY



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
print("Lorenz96")
K=36
J=10
t=np.arange(0.0,10.0,0.01)
z=np.zeros((9,ITER))
for i in range(0,ITER):
    print(i)
    x0=np.random.rand(K*(J+1))
    z[0,i]=1
    z[1,i]=10
    z[2,i]=np.exp(np.random.normal(loc=2.0,scale=0.1))
    z[3,i]=10
    z[4:,i]=output(odeint(Lorenz96,x0,t,tuple(z[:4,i])),1000,100)

y_pred=gp.predict(np.transpose(z[:4,:]))

#X2=np.append(X,np.transpose(z[:4,:]),axis=0)
#Y2=np.append(Y,np.transpose(z[4:,:]),axis=0)
#np.save("parameters",X2)
#np.save("Output",Y2)
zt=np.transpose(z[4:,:])
diff=y_pred-zt
sqdiff=diff**2
print(np.sum(diff))
print(np.sum(sqdiff))

