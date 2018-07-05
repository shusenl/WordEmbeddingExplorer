import numpy as np
import scipy
import math

def pairsEuclideanAnalogyQualityMeasures(dataMatrix, pairs=None):
        #print 'At embeddingQualityMeasure -> input data matrix:', dataMatrix.shape
        if pairs is None:
            pairs = [[x*2, x*2+1] for x in range(dataMatrix.shape[1]/2)]
            # always assume to be even, default pair [[0,1],[2,3],..]
        error = 0.0
        count = 0
        for i, pair_i in enumerate(pairs):
            for j, pair_j in enumerate(pairs):
                if (i != j) and (i<j):
                   count += 1
                   #error += _enclideanAnalogyTest(dataMatrix[:,pair_i[0]],
                   #error += _cosineAnalogyTest(dataMatrix[:,pair_i[0]],
                   error += _pairAngleAnalogyTest(dataMatrix[:,pair_i[0]],
                                                  dataMatrix[:,pair_i[1]],
                                                  dataMatrix[:,pair_j[0]],
                                                  dataMatrix[:,pair_j[1]])

        if math.isnan(error):
            print error
        return error

def _enclideanAnalogyTest(pair_i1, pair_i2, pair_j1, pair_j2):
    return np.linalg.norm(pair_i1 - pair_i2 + pair_j2 - pair_j1)

def _pairAngleAnalogyTest(pair_i1, pair_i2, pair_j1, pair_j2):
    if (np.linalg.norm(pair_i1-pair_i2)==0.0) or (np.linalg.norm(pair_j1-pair_j2)==0.0):
        return 0.0
    else:
        return scipy.spatial.distance.cosine(pair_i1-pair_i2, pair_j1-pair_j2)

def _cosineAnalogyTest(pair_i1, pair_i2, pair_j1, pair_j2):
    return scipy.spatial.distance.cosine(pair_i1-pair_i2+pair_j2, pair_j1)
