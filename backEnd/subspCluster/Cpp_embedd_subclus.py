import numpy as np
from scipy import stats
from sklearn.cluster import SpectralClustering
from subsp import *
from common import *
from scipy import linalg
#import matplotlib.pyplot as plt

#the naive mapping algorithm
def matchMap(Ref, L):
    print "### starting match map ###"
    newL = L
    if len(Ref)!= len(L):
        print("ERROR: The two label vectors must be of the same length")
    else:
        LabRef = np.unique(Ref)
        nCls1 = len(LabRef)
        Lab = np.unique(L)
        labMatch = LabRef;
        if len(LabRef) == len(Lab):
          for label in LabRef:
            labelIndex = np.where(Ref==label)
            elements = np.squeeze(np.take(L, labelIndex))
            #print elements
            labMatch[label] = np.argmax(np.bincount(elements))

          print "map:", labMatch

          for i in range(0, len(L)):
            newL[i] = np.squeeze(np.where(labMatch==L[i]))
    return newL


# the input are all numpy array
def Cpp_embedd_subspaceClustering(data, K, grMethod, gndLab):
#def Cpp_embedd_subspaceClustering(data, K, grMethod):

    K = int(K)
    grMethod = int(grMethod)

    data = np.matrix(data)

    print "data:", np.shape(data)
    print "K:", K, np.shape(K)
    print "grMethod", np.shape(grMethod)

    [dim, nPoints] = data.shape;
    print "dim:", dim, "nPoints:", nPoints

    #grMethod = int(argv[3]);
    #if len(sys.argv) == 5:
    #    gndLab, temp1, temp2 = loadData(argv[4]);
    #    flag = 1 # is ground truth given?
    #else:
    #    flag = 0

    # is ground truth given?
    #np.transpose(gndLab)
    gndLab = gndLab.astype(int)
    print "gndLab:", np.shape(gndLab)
    [labelPoints,labelDim] = gndLab.shape
    print "labelDim:", labelDim, "labelPoints:", labelPoints

    if(labelPoints == nPoints and labelDim == 1):
      flag = 1
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
    #Grps = np.add(Grps, 1)
    print "Grps:", np.shape(Grps), type(Grps),"\n", Grps

    if flag == 1:
      print "\nuse best map"
      gndLab = np.asarray(np.asmatrix(gndLab).transpose())
      gndLab = np.add(gndLab, -1)
      print "gndLab:", np.shape(gndLab), type(gndLab),"\n", gndLab

      #Grps = matchMap(gndLab,Grps)
      Grps = bestMap(np.transpose(gndLab),Grps)

      Grps = np.squeeze(np.asarray(Grps.transpose()))
      #Grps = np.add(Grps,-1)
      print "Grps:", np.shape(Grps), type(Grps),"\n"
      #np.transpose(Grps)

    #convert to row-major for consistancy
    if(Grps.flags['C_CONTIGUOUS'] == False):
      Grps = np.ascontiguousarray(Grps)

    print "\nOutput final label\n",Grps

    print "clusterLabel shape:", Grps.shape
    print "affinity shape:", Z.shape

    return [np.matrix(Grps.astype(np.float64)), Z]
