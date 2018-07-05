from sys import argv
import sys
import numpy as np
from scipy import stats
from sklearn.cluster import SpectralClustering
from subsp import *
from common import *
from scipy import linalg
import matplotlib.pyplot as plt


if __name__ == '__main__':
    [data,nPoints,dim] = loadData(argv[1]);

    K = int(argv[2]);
    grMethod = int(argv[3]);
    if len(sys.argv) == 5:
        gndLab, temp1, temp2 = loadData(argv[4]);
        flag = 1 # is ground truth given?
    else:
        flag = 0
   
    #standardize = 0
    #if standardize == 1:
    #    data = stats.zscore(data, 1, 0);

    if grMethod == 1: # SSC
        alpha = 0.05;
        C = sparsecoding(data,alpha,nPoints)
        Z = BuildAdjacency(C)         
    elif grMethod == 2: #LRR
        Q = linalg.orth(np.transpose(data))
        A = np.dot(data,Q)
        lam = 0.1
        Z,E = lrrA(np.matrix(data),np.matrix(A),lam)
        Z = np.dot(Q,Z)
        Z = np.abs(Z) + np.abs(np.transpose(Z))
    elif grMethod == 3: #L2
        ktrunc = 15
        Z = L2encode(data,ktrunc)
        Z = np.abs(Z) + np.transpose(np.abs(Z))
    else:
        print('Not a valid option');
        exit();
    
    # Spectral Clustering
    Grps = normspecclus(Z,K)

    print "\nbefore Grps:", Grps
    if flag == 1:
        Grps = bestMap(np.transpose(gndLab),Grps)
    print "\nafter Grps:", Grps
    print "\ngndLab:", gndLab
    
    plt.figure()
    #plt.scatter(np.arange(nPoints),Grps,s = 30, c = 'r')
        
    subspList = [];
    for i in range(0,K):
        ids = np.where(Grps == i)[0]
        gr = Z[np.ix_(ids,ids)];
        outIds = np.where(Grps != i)[0]
        outgr = Z[np.ix_(outIds,outIds)];
        obj = subsp(data[:,ids],gr,data[:,outIds],outgr)
        obj.estDim()
        #normalize the column
        np.apply_along_axis(np.linalg.norm, 0, obj.basis)
        subspList.append(obj)
    
    for obj in subspList:
        print obj.dim, obj.pcadim
        print obj.basis
        print "basis:", obj.basis.shape, "data:", data.shape
        projectedData =  np.matrix(obj.basis.transpose()) * np.matrix(data)
        print "Project data:", projectedData.shape
        plt.scatter(projectedData[0,:], projectedData[1,:])
        plt.show()
    
    #plt.show()
    
            



