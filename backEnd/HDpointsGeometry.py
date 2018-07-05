import numpy as np
from pyhull.convex_hull import ConvexHull
from pyhull import qconvex
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors.kde import KernelDensity
from sklearn.metrics import pairwise_distances
from random import gauss, seed
import pdb

def computeKernelDensityEstimation(dataMatrix, kernelType, kernelSize):
  kde = KernelDensity(kernel = kernelType,
                      bandwidth = kernelSize).fit(dataMatrix)
  return kde.score_samples(dataMatrix)

def computeConvexHullVolume(points):
  outputStr = qconvex("FA", points)
  volumeSize = outputStr[-2].split()[3]
  print "ConvexHullSize:",volumeSize
  return volumeSize

#points = np.random.rand(30, 5)
#computeConvexHullVolume(points)
def make_rand_vector(dims):
    vec = [gauss(0, 1) for i in range(dims)]
    mag = sum(x**2 for x in vec) ** .5
    return [x/mag for x in vec]

def computeClusterDiameter(dataMatrix, distance):
  pairDistances = []
  for x in dataMatrix:
    for y in dataMatrix:
      pairDistances.append(distance(x,y))
  return max(pairDistances)

def computeClusterBinearyLinearSeparation(dataMatrix, label, hardness):
  clf = LinearSVC(C=hardness)
  print "SVM input data shape", dataMatrix.shape
  clf.fit(dataMatrix, label)
  classification = clf.predict(dataMatrix)
  #compare classification with label
  distance = clf.decision_function(dataMatrix)
  directionVec = clf.coef_
  #print distance
  return distance, classification, directionVec

def orthogonalizeToVec(source, target):
  vecOrth = source - target*np.dot(source, target)
  #print "vecOrth:", vecOrth.shape
  vecOrthNorm = vecOrth/np.linalg.norm(vecOrth)
  return vecOrthNorm

def computeRotationAlignToVector(directionVec):
  #print "directionVec:", directionVec.shape
  #normalize
  directionVec = directionVec/np.linalg.norm(directionVec)

  I = np.identity(directionVec.shape[0])
  unit = np.asarray(np.eye(1, directionVec.shape[0], 0))
  #print I, I.shape, unit, unit.shape, directionVec
  u = directionVec - unit
  u = u / np.linalg.norm(u) #normalize
  #print u, u.shape
  rotation = I - 2*u*np.transpose(u)
  #print '2uu_T:', rotation.shape
  #print 'reflection det:', np.linalg.det(rotation)
  return rotation.tolist()
def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def computeOptimalAnalogyProjection(dataMatrix, label, optimizationType):
  projMat = []
  clf = LinearSVC()
  #print "SVM input data shape", dataMatrix.shape
  clf.fit(dataMatrix, label)
  #classification = clf.predict(dataMatrix)
  directionVec = np.squeeze(np.asarray(np.transpose(clf.coef_)))
  #create reflection matrix rotate this vector to [1,0,...0]
  I = np.identity(directionVec.shape[0])
  unit = np.asarray(np.eye(1, directionVec.shape[0], 0))
  #print I, I.shape, unit, unit.shape, directionVec
  u = directionVec - unit
  u = u / np.linalg.norm(u) #normalize
  #print u, u.shape
  rotation = I - 2*u*np.transpose(u)
  #print '2uu_T:', rotation.shape
  print 'reflection det:', np.linalg.det(rotation)
  #print "label", label
  leftWordsLabelIndex = [i for i, x in enumerate(label) if x == 0]
  #print "leftWordsLabel:", leftWordsLabelIndex
  analogyWordLeft = dataMatrix[leftWordsLabelIndex,:]
  embedding = []
  #print "analogyWordLeft", analogyWordLeft.shape
  #print "dataMatrix:", dataMatrix.shape
  #print "optimizationType:", optimizationType

  if optimizationType == "SVM+PCA":
   #compare classification with label
    #distance = clf.decision_function(dataMatrix)
    #find the other direction
    #use random vector
    #vec2 = np.array(make_rand_vector(directionVec.size))
    #use 1D PCA
    pca = PCA(n_components=1)
    pca.fit(analogyWordLeft)
    #print 'analogyWordLeft:', analogyWordLeft.shape
    #analogyWordLeftNoDuplicate = unique_rows(analogyWordLeft)
    #print "analogyWordLeftNoDuplicate", analogyWordLeftNoDuplicate.shape
    #pca.fit(analogyWordLeftNoDuplicate)
    vec2 = pca.components_
    #print "directionVec:", directionVec.shape
    #print "vec2:", vec2.shape

    #make vec2 orthonormal to directionVec
    #print "check angle between directionVec and PCA 1D:", np.dot(vec2, directionVec)
    vecOrthNorm = orthogonalizeToVec(vec2, directionVec)
    #vecOrth = vec2 - directionVec*np.dot(vec2, directionVec)
    #print "vecOrth:", vecOrth.shape
    #vecOrthNorm = vecOrth/np.linalg.norm(vecOrth)
    projMat = np.vstack((vecOrthNorm, directionVec))
    #print projMat.shape, '\n', projMat
  elif optimizationType == "LDA":
    # lda = LinearDiscriminantAnalysis(n_components=2)
    lda = LinearDiscriminantAnalysis(n_components=2)#, solver='lsqr')
    # print len(label), dataMatrix.shape
    lda.fit(dataMatrix, np.asarray(label))
    embedding = lda.transform(dataMatrix)
    # print embedding.shape
    # embedding = embedding.tolist()
    # embedding = lda.fit_transform(dataMatrix, label).tolist()
    # print embedding
    projMat = lda.scalings_
    # print "scalings_ shape:", lda.scalings_.shape

  elif optimizationType == "PCA":
    pca = PCA(n_components=2)
    #pca = decomposition.PCA(n_components='mle')
    pca.fit(dataMatrix)
    #print pca.explained_variance_ratio_
    #print sum(pca.explained_variance_ratio_)
    #print 'number of component:'
    #print pca.n_components_
    # embedding = pca.transform(dataMatrix)
    projMat = pca.components_

  elif optimizationType == "SVM+REG":
    from sklearn.svm import SVC, SVR
    import scipy.linalg as sl
    pairID = np.array([ [i*2,i*2+1] for i in range(len(label)/2)])
    #print 'pairID:', pairID.shape, pairID
    n_features = dataMatrix.shape[1]
    clf = SVC(kernel = 'linear')
    clf.fit(dataMatrix,label)
    directionVec1 = clf.coef_.reshape((n_features,1))
    directionVec1 = directionVec1/np.linalg.norm(directionVec1)

    from sklearn.linear_model import Ridge
    #reg = Ridge(1e-4)
    reg = SVR(kernel='linear')
    D = []
    dval = []

    for i in range(pairID.shape[0]):
        D.append(dataMatrix[pairID[i,0],:] - dataMatrix[pairID[i,1],:])
        dval.append(0)
        #pdb.set_trace()
        #print dataMatrix[pairID[i,0],:].reshape(1,-1).shape
        #print dataMatrix[pairID[:,0],:].shape
        dists0 = pairwise_distances(dataMatrix[pairID[i,0],:].reshape(1,-1)
                                              ,dataMatrix[pairID[:,0],:])
        rank0 = np.argsort(dists0)
        dists1 = pairwise_distances(dataMatrix[pairID[i,1],:].reshape(1,-1)
                                                  ,dataMatrix[pairID[:,1],:])
        rank1 = np.argsort(dists1)

        for j in range(2):
            ind = np.random.randint(pairID.shape[0])
            while ind == i:
                ind = np.random.randint(pairID.shape[0])

            D.append(dataMatrix[pairID[i,0],:] - dataMatrix[pairID[ind,0],:])
            #dval.append(np.where(np.squeeze(rank0) == ind)[0][0])
            dval.append(dists0[0,ind])
            D.append(dataMatrix[pairID[i,1],:] - dataMatrix[pairID[ind,1],:])
            dval.append(dists1[0,ind])
            #dval.append(np.where(np.squeeze(rank1) == ind)[0][0])

    #print np.array(D).shape
    #print np.array(D).shape
    reg.fit(np.squeeze(np.array(D)), np.squeeze(dval))
    directionVec2 = reg.coef_.reshape((n_features,1))
    directionVec2 = directionVec2/np.linalg.norm(directionVec2)
    directionVec2 = directionVec2 - directionVec1*np.dot(directionVec2.T, directionVec1)
    directionVec2 = directionVec2/np.linalg.norm(directionVec2)
    projMat = np.concatenate((directionVec1,directionVec2),axis=1).T

  elif optimizationType == "SVMplane":
    pca = PCA(n_components=2)
    pca.fit(analogyWordLeft)
    vec1 = orthogonalizeToVec(pca.components_[0], directionVec)
    vec2 = orthogonalizeToVec(pca.components_[1], directionVec)
    vec1 = orthogonalizeToVec(vec1, vec2)
    #print "vec1 dot vec2:", np.dot(vec1, vec2)
    projMat = np.vstack((vec1, vec2))

  else:
    print "Can't find OptimizationType:", optimizationType

  return projMat, rotation.tolist()
  # return projMat, rotation.tolist(), embedding
