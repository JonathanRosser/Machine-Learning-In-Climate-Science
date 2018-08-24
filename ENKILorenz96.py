import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


#Create an ensemble, define the first M terms of the vector as the parameters and the following N terms of the vector as the data ( G(u) )
M=4
N=5
#ztrue contains the true parameters and the observed data (y) calculated from them which we are using to train the model.
ztrue=np.zeros((N+M))
#Define the true parameters
#for i in range(0,M):
#    ztrue[i]=np.random.uniform()
ztrue[0]=1 #h 
ztrue[1]=10 #F
ztrue[2]=10 #c
ztrue[3]=10 #b
K=36 #Number of slow variables 
J=10 #Number of fast variables per slow variable
longspin=1000
fastspin=100
longtime=10000
fasttime=1000



#Define functions for use in the Lorenz model evaluations

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

#Conduct an initial long run to determine the true data

t=np.arange(0.0,longtime/100,0.01)
x0=np.random.rand(K*(J+1))
print("solve ODE")
x=odeint(Lorenz96,x0,t,tuple(ztrue[0:M]))
print("Calcuate true data")
ztrue[M:M+N]=output(x,longtime,longspin)
#Define the covariance
print("calculate Covariance")
r=0.0001
Gamma=(r**2)*np.diag((np.var(np.mean(x[longspin:,:K],1)),np.var(np.mean(x[longspin:,K:],1)),np.var(np.mean(np.square(x[longspin:,:K]),1)),np.var(np.mean(np.square(x[longspin:,K:]),1)),np.var(XY(x,longtime,longspin))))

#Output the data
Q=400 #Number of ensemble members
ITER=10 #Number of iterations
parameters=np.zeros((4,Q*ITER))
Output=np.zeros((5,Q*ITER))

#Prediction step

#Create an initial ensemble, Q is the number of ensemble members
z=np.zeros((N+M,Q))
for i in range(0,M):
    if i == 0:
        for j in range(0,Q):
            print(j)
            z[i,j]=np.random.normal(loc=0.0,scale=1.0)
    if i ==1:
        for j in range(0,Q):
            print(j)
            z[i,j]=np.random.normal(loc=10.0,scale=10.0)
    if i ==2:
        for j in range(0,Q):
            print(j)
            z[i,j]=np.exp(np.random.normal(loc=2.0,scale=0.1))
    if i==3:
        for j in range(0,Q):
            print(j)
            z[i,j]=np.random.normal(loc=5,scale=10.0)



t=np.arange(0.0,fasttime/100,0.01)

count=0

while count<ITER:
            #Prediction step

            #Calculate the new ensemble predictions

            for i in range(0,Q):
                x0=np.random.rand(K*(J+1))
                z[M:M+N,i]=output(odeint(Lorenz96,x0,t,tuple(z[0:M,i])),fasttime,fastspin)
                #Update the output data
                parameters[:,count*Q+i]=z[:M,i]
                Output[:,count*Q+i]=z[M:M+N,i]
                print(count,i)



            #Sample mean
            zbar=np.mean(z,axis=1)
            print("zbar= ",zbar)
            if zbar[M]==0 and zbar[M+1]==0 and zbar[M+2]==0:
                print("Solver has failed")
                break
            #Sample covariance
            #Mean of outer products
            MOP=np.zeros((N+M,N+M))
            for i in range(0,Q):
                MOP=MOP+np.outer(z[:,i],np.transpose(z[:,i]))/Q
            #Covariance
            C=MOP-np.outer(zbar,np.transpose(zbar))






            #Analysis step
            print("Analysis")
            #Define the Kalman gain

            H=np.append(np.zeros((N,M)),np.identity(N),axis=1)
            A=np.matmul(C,np.transpose(H))
            B=np.matmul(H,np.matmul(C,np.transpose(H)))+Gamma
            KAL=np.transpose(np.linalg.solve(np.transpose(B),np.transpose(A)))


            #Generate random data

            yrand=np.zeros((N,Q))
            for i in range(0,Q):
                yrand[:,i]=ztrue[M:M+N]+np.random.normal(loc=0.0,scale=np.sqrt(np.diag(Gamma)))


            #Update the ensemble members
            z=z-np.matmul(KAL,np.matmul(H,z))+np.matmul(KAL,yrand)







            #Convergence
            #Compute the mean of the parameter update:
            u=np.mean(z,axis=1)[0:M]
            print("u=",u)

            #Check for convergence:
            tau=1
            #SHOULD THIS BE -1 or -0.5 for the power of the matrix
            #alpha=np.linalg.norm(np.matmul(np.linalg.matrix_power(Gamma,-1)[M:N+M,M:N+M],ztrue[M:N+M]-np.matmul(G,u)))
            #beta=tau*np.linalg.norm(np.matmul(np.linalg.matrix_power(Gamma,-1)[M:N+M,M:N+M],etadag))
            #if alpha <= beta:
            #    break
            count += 1

np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/ENKIparameters400x4ITER10.npy",parameters)
np.save("/home/jprosser/Documents/WorkPlacements/Caltech/Data/ENKIOutput400x4ITER10.npy", Output)



