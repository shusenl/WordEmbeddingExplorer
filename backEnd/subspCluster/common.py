import numpy as np
import spams
from scipy import linalg
from sklearn.cluster import *
from munkres import Munkres

def loadData(filename):

    inp = open(filename,'r')
    line = inp.readline().split()
    nPoints = int(line[0])
    dim = int(line[1])
            
    line = inp.readline().split()
    data = np.asanyarray(line,dtype=float)
    data = data.reshape([nPoints,dim])
    data = np.transpose(data)

    return data,nPoints,dim 

def L2encode(X,ktrunc):
    d,nPoints = X.shape
    X = np.matrix(X)
    J = np.matrix(linalg.inv(np.transpose(X)*X + 1e-6*np.eye(nPoints)))
    Z = np.zeros((nPoints,nPoints))
    
    for i in range(0,nPoints):
        xi = np.matrix(X[:,i])
        ei = np.matrix(np.zeros((nPoints,1)))
        ei[i] = 1
        term2 = (np.transpose(ei)*J*np.transpose(X)*xi)/(np.transpose(ei)*J*ei) 
        ci = J*(np.transpose(X)*xi - ei*term2)
        ci = np.array(ci/linalg.norm(ci,2))
        tind = np.argsort(-np.abs(ci))
        
        ci[tind[ktrunc:nPoints]] = 0
        Z[:,i] = ci[:,0]        
        
    return Z

#### Cpp interface ######
def Cpp_L2encode(X):
    print "L2 encode"
    Z = np.array(L2encode(X, ktrunc = 15)) 
    Z = np.abs(Z) + np.transpose(np.abs(Z))    
    return [ Z ]

def sparsecoding(data,alpha,nPoints):

    C = np.zeros((nPoints,nPoints),np.float64);
    d = data.shape[0]
    for i in range(0,nPoints):
        D = data;
        D = np.delete(D,i,1);
        x = np.asfortranarray(np.reshape(data[:,i],(d,1)))
        D = np.asfortranarray(D)
        codes = spams.lasso(x,D=D, lambda1=alpha, pos=True)
        codes = codes.A
        ind = range(0,i) + range(i+1,nPoints);
        for j in range(0,nPoints-1):
            C[ind[j],i] = codes[j]; 
    return C

#### Cpp interface ######
def Cpp_sparsecoding(data):
  d, nPoints = data.shape
  print "Cpp_sparsecoding", "pointCount:", nPoints, "Dimension:", d
  alpha = 0.05;
  #need to run build adjacency after sparse coding
  return [ BuildAdjacency(sparsecoding(data, alpha, nPoints)) ]

def BuildAdjacency(CMat):
    N = CMat.shape[0];
    CAbs = np.abs(CMat);
    for i in range(0,N):
        c = CAbs[:,i];
        mVal = np.amax(c);
        if mVal != 0:
            CAbs[:,i] = CAbs[:,i]/mVal;
    CSym = CAbs + np.transpose(CAbs);
    
    return CSym

    
def lrrA(X,A,lam):
    m = A.shape[1]
    d,n = X.shape
    tol = 1e-8
    maxIter = 1e6
    rho = 1.1  
    max_mu = 1e10
    mu = 1e-6
    atx = np.transpose(A)*X
    inv_a = linalg.inv(np.transpose(A)*A + np.eye(m))
    J = np.zeros((m,n))
    Z = np.zeros((m,n))
    E = np.zeros((d,n))
    Y1 = np.zeros((d,n))
    Y2 = np.zeros((m,n))
    iter = 0
    
    while iter<maxIter:
        iter = iter+1
        # Update J
        temp = Z + Y2/mu
        U,sigma,V = linalg.svd(temp)
        V = np.transpose(V)
        svp = np.sum(sigma > (1/mu))
        
        if svp == 0:
            svp = 1
            sigma = 0
        else:
            sigma = sigma[range(0,svp)] - (1/mu)
        if svp > 1:
            sigma = np.matrix(np.diag(sigma))
        
        J = np.matrix(U[:,np.arange(0,svp)])*sigma*np.matrix(np.transpose(V[:,range(0,svp)]))
        Z = atx - np.dot(np.transpose(A),E) + J + (np.dot(np.transpose(A),Y1) - Y2)/mu
        Z = np.dot(inv_a,Z)
        
        xmaz = X - A*np.matrix(Z)
        temp = xmaz + Y1/mu
        E = solve_l1l2(temp,lam/mu)
        leq1 = xmaz-E
        leq2 = Z-J
        stopC = np.max([np.max(np.abs(leq1)),np.max(np.abs(leq2))])
        
        if stopC < tol:
            break
        else:
            Y1 = Y1 + mu*leq1
            Y2 = Y2 + mu*leq2
            mu = np.min([max_mu,mu*rho])
        
    return Z, E

def solve_l1l2(W,lam):
    n = W.shape[1]
    E = W
    for i in range(0,n):
        E[:,i] = solve_l2(W[:,i],lam)
    return E

def solve_l2(w,lam):
    nw = linalg.norm(w)
    if nw > lam:
        x = (nw - lam)*w/nw
    else:
        x = w*0
    return x 

#### Cpp interface ######
def Cpp_lrrA(data):
  print "Cpp_LRR", "data:", data.shape
  Q = linalg.orth(np.transpose(data))
  A = np.dot(data,Q)
  lam = 0.1
  Z,E = lrrA(np.matrix(data),np.matrix(A),lam)
  Z = np.dot(Q,Z)
  Z = np.abs(Z) + np.abs(np.transpose(Z))
  print Z
  return [ np.array(Z) ]

def normspecclus(Z,K):
    nPoints = Z.shape[0]
    Z = np.array(Z)
    D = np.diag(np.power(np.sum(Z,1),-0.5))
    D[np.isnan(D)] = 0
    D[np.isinf(D)] = 0
    Lap = np.eye(nPoints) - np.matrix(D)*np.matrix(Z)*np.matrix(D)
    uKS,sKS,vKS = linalg.svd(Lap)
    vKS = np.transpose(vKS)
    f = vKS.shape[1]
    ker = vKS[:,range(f-K,f)]
    for i in range(0,nPoints):
        ker[i,:] = ker[i,:]/linalg.norm(ker[i,:])
    kmobj = KMeans(n_clusters = K)
    L = kmobj.fit_predict(ker)
    
    return L

def bestMap(L1,L2):
    print "### starting best map, both index need to start with 0 !!###"
    if len(L1)!= len(L2):
        print("ERROR: The two label vectors must be of the same length")
        newL2 = L2
    else:
        L2 = np.reshape(L2, (len(L2),1))
        Lab1 = np.unique(L1)
        nCls1 = len(Lab1)
        Lab2 = np.unique(L2)
        nCls2 = len(Lab2)
        print "Lab1:", Lab1, "Lab2", Lab2, "nCls1:", nCls1, "nCls2:", nCls2
        
        nCls = np.max([nCls1, nCls2])
        G = np.zeros((nCls,nCls));
        for i in range(0,nCls1):
            for j in range(0,nCls2):
                dec1 = L1==Lab1[i]
                dec2 = L2==Lab2[j]
                #print "\ndec1:", dec1, "\ndec2:", dec2
                if np.sum(dec1)>0 and np.sum(dec2)>0:
                    dec = np.sum(np.bitwise_and(dec1,dec2))
                    G[i,j] = dec
        m = Munkres()
        indexes = m.compute(-G)

        print "index:", indexes
        c = np.zeros(nCls)
        for i in range(0,nCls):
            c[ indexes[i][1] ] = indexes[i][0]
        print "c map:", c    

        newL2 = L2*0;
        for i in range(0,nCls2):
            newL2[L2 == Lab2[i]] = Lab1[c[i]];
    
    return newL2
    
