#################
# data the data matrix
# Grps clustering result 
  #(labels : array, shape [n_samples,] Index of the cluster each sample belongs to.)
# Z the affinity matrix
# K number of cluster
#####################

#import sys
import numpy as np
from scipy import stats
from subsp import *
#from common import *
from scipy import linalg
#import matplotlib.pyplot as plt

# the input are all numpy array
def basisEstimation(data, Grps, Z, K):
    
    K = int(K)
    data = np.matrix(data)
    Z = np.matrix(Z)
    Grps = Grps.astype('int')
    
    print "data:", np.shape(data)    
    print "Z:", np.shape(Z)    
    print "K:", K, np.shape(K) 
    print "Grps", np.shape(Grps),"!!!!shape:(sample,1)"
    
    #print Grps

    #Grps is an array
    
    #plt.figure()
    #plt.scatter(np.arange(nPoints),Grps,s = 30, c = 'r')
        
    subspList = [];
    for i in range(0,K):
        ids = np.where(Grps == i)[0]
        print "ids:",ids
        gr = Z[np.ix_(ids,ids)];
        outIds = np.where(Grps != i)[0]
        outgr = Z[np.ix_(outIds,outIds)];
        obj = subsp(data[:,ids],gr,data[:,outIds],outgr)
        obj.estDim()
        
        #print "Basis", obj.basis.flags
        if(obj.basis.flags['C_CONTIGUOUS'] == False):
          obj.basis = np.ascontiguousarray(obj.basis)
        #normalize the column
        #np.apply_along_axis(np.linalg.norm, 0, obj.basis)
        #print "Dim:", obj.dim, "Shape:", np.shape(obj.basis)
        #print obj.basis
        subspList.append(obj)
  
    subDimList = []    
    for i in range(0,K):
        obj = subspList[i]
        print "Basis:", obj.dim, np.shape(obj.basis)
        #print obj.basis, obj.basis.flags
        subDimList.append(obj.basis)
    #return a tuple
    return subDimList
    #plt.show()
