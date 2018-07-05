from estimateRandomProjPDF import *
from mongoStorage import *
import matplotlib.pyplot as plt
import pylab

pairCounts = [23, 116, 30, 68, 23, 32, 29, 37, 34, 33, 41, 40, 37, 30]
#pairCounts = [69, 24, 33, 30, 38, 35, 34, 42, 41, 38, 31]
wordCounts = [i*2 for i in pairCounts]

def errorPDFpreComputation(sampler, dataDim, wordCount, projMethod, sampleCount=1000, embeddingDim=2):
    sampleQuery = {}
    sampleQuery['dataDim'] = dataDim
    sampleQuery['wordCount'] = wordCount
    sampleQuery['projMethod'] = projMethod
    mongoDB = mongoStorage()

    errorMean, histo = pairProjErrorPDF(sampler, projMethod, pairsEuclideanAnalogyQualityMeasures, embeddingDim, sampleCount, wordCount)
    #save to database for later
    sampleQuery['errorMean']=errorMean
    sampleQuery['histo']=histo
    mongoDB.save(sampleQuery)

    return errorMean

def main():
    sampler = wordEmbeddingSampler('Glove','300d', 300);
    #for wordCount in wordCounts:
        #errorPDFpreComputation(sampler, 300, wordCount, 'SVM+REG')
        #errorPDFpreComputation(sampler, 300, wordCount, 'PCA')
        #errorPDFpreComputation(sampler, 300, wordCount, 'SVM+PCA')
    '''
    SVM_REG = []
    SVM_PCA = []
    PCA = []
    dims = [x*20+2 for x in range(1,15)]
    sampleCount = 1000
    for dim in dims:
       sampler.setDim(dim)
       errorMean1 = errorPDFpreComputation(sampler, dim, 60, 'SVM+REG', sampleCount)
       errorMean2 = errorPDFpreComputation(sampler, dim, 60, 'SVM+PCA', sampleCount)
       errorMean3 = errorPDFpreComputation(sampler, dim, 60, 'PCA', sampleCount)
       SVM_REG.append(errorMean1)
       SVM_PCA.append(errorMean2)
       PCA.append(errorMean3)
       #Y=[387.1371199, 129.519062071,50.293665722,57.7206883598,56.4200539905,53.4907481048,54.0695607061]
       #X=[10, 50, 100, 150, 200, 250, 300]
    pylab.plot(dims, SVM_REG, '-r', label='SVM_REG')
    pylab.plot(dims, SVM_PCA, '-g', label='SVM_PCA')
    pylab.plot(dims, PCA, '-b', label='PCA')
    pylab.legend(loc='upper left')
    pylab.show()
    '''
    sampleCount = 2000
    for pairCount in pairCounts:
       errorPDFpreComputation(sampler, 300, pairCount*2, 'SVM+REG', sampleCount)
       errorPDFpreComputation(sampler, 300, pairCount*2, 'SVM+PCA', sampleCount)
       errorPDFpreComputation(sampler, 300, pairCount*2, 'PCA', sampleCount)


if __name__ == "__main__":
    main()
