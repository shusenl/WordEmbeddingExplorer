from word2VecHelper import * #for querying word2Vec
from HDpointsGeometry import *
from embeddingQualityMeasures import *
import numpy as np
import random

#for debug
import matplotlib.pyplot as plt

class wordEmbeddingSampler(object):
    def __init__(self, dbName='Glove', collectionName='300d', dim=300):
        self.dbName = dbName
        self.dim = dim
        self.collectionName = collectionName
        self.counter = 0
        self.wordVecList = []
        self._initSampleBuffer()

    def _initSampleBuffer(self):
        self.wordVecList[:] = []
        self.counter = 0
        self.wordVecList = getVectorsFromWordEmbeddingDatabase(self.dbName, self.collectionName)
    def setDim(self,dim):
           self.dim = dim
    def getNext(self):#return the next random vector
        #if self.counter<len(self.wordVecList):
           randIndex = int(random.random()*(len(self.wordVecList)-1))
           #print randIndex
           wordVec = self.wordVecList[randIndex]
           #self.counter += 1
           if len(wordVec)>self.dim:
               #bootstrap
               selectDim = random.sample(range(len(wordVec)), self.dim)
               return [wordVec[i] for i in selectDim]
           else:
               return wordVec
        #else:
        #   self._initSampleBuffer()
        #   return self.getNext()

class gaussRandomSampler(object):
    def __init__(self, dim):
        self.dim = dim;
    def getNext(self):
        wordVec = np.random.normal(size=self.dim).tolist()
        #print wordVec
        return wordVec

class uniformRandomSampler(object):
    def __init__(self, dim):
        self.dim = dim;
    def getNext(self):
        wordVec = [random.random() for i in range(self.dim)]
        #print wordVec
        return wordVec


############################################################################
def projFunc(dataMatrix, projDim, optimizationType="SVM+PCA"):
    count = dataMatrix.shape[0]
    label = [i%2 for i in range(count)] #generate 0, 1, 0, 1 pattern
    #print dataMatrix.shape, label
    projMat, rotation = computeOptimalAnalogyProjection(dataMatrix, label, optimizationType)
    #print dataMatrix.shape
    #print projMat
    #print projMat.shape
    embedding = dataMatrix*np.transpose(projMat)
    #print 'embedding:',embedding.shape

    '''
    for i in xrange(embedding.shape[0]/2):
        plt.plot(embedding[[i*2, i*2+1], 0].tolist(), embedding[[i*2, i*2+1], 1].tolist())
    plt.show()
    '''

    return embedding


############################################################################
def pairProjErrorPDF(randomSampler, projMethod, errorFunc, dim, sampleCount, wordCount):
        errors = []

        for i in range(sampleCount):
            wordVecMatrix = []
            for j in range(wordCount):
                wordVecMatrix.append(randomSampler.getNext())
            embedding = projFunc(np.matrix(wordVecMatrix), dim, projMethod)
            errors.append(errorFunc(np.transpose(embedding)))
        errorMean = np.mean(errors)
        print 'wordCount:', wordCount,', dim:', randomSampler.dim, ', error:',errorMean
        histo = np.histogram(errors, bins=20)
        return errorMean, [histo[0].tolist(),histo[1].tolist()]

        #return errors
        #exit()
        #compute histogram
        #hist = np.histogram(errors, bins='auto')
        #plt.hist(errors, bins='auto')  # plt.hist passes it's arguments to np.histogram
        #plt.hist(errors)  # plt.hist passes it's arguments to np.histogram
        #plt.title("Histogram of random data error")
        #plt.show()


def bootstrapProjErrorPDF(data, projFunc, errorFunc, bootstrapDim, sampleCount):

        #loop through to generate histogram
        dataDim = data.shape[0] #number of row is the dim
        errors = []

        for i in range(sampleCount):
            #select bootstrapdim from data dim
            selectDim = random.sample(range(dataDim), bootstrapDim)
            bootstrapData = data[selectDim, :] # select subset of dimesions
            embedding = projFunc(bootstrapData, dim=2)
            errors.append(errorFunc(embedding)) # assume data store word in pair in a linear order

        errorMean = np.mean(errors)
        #print errorMean
        return errorMean

        #plt.hist(errors)  # plt.hist passes it's arguments to np.histogram
        #plt.title("Histogram of bootstrap data error")
        #plt.show()
        #return errors


        #estimate projection quality based on random projection
        #compute projection quality measure
