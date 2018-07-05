from sklearn import decomposition
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.manifold import TSNE

from pymongo import MongoClient

#from Cpp_embedd_subclus import *
#from Cpp_embedd_subsp import *
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import pairwise_distances

from sets import Set
import json
import operator

from scipy import spatial
import random

#####!!! do not use any global variables here !!!!#####

############################## python functions #############################

######################################################
def tSNEembedding(data, components):
    model = TSNE(n_components=components, random_state=0)
    return model.fit_transform(data)

######################################################
def clusterNumFromLabel(label):
    #label need to start from 0
    return len(set(label)) - (1 if -1 in label else 0)
    #return max(Set(label.tolist()))+1

######################################################
def computeClusterDistMatrix(data, label, clusterProcessFunc, distFunc):
    numCluster = clusterNumFromLabel(label)
    print 'clusterNum: ', numCluster
    distMatrix = np.zeros([numCluster, numCluster])
    if numCluster == 1:
        return distMatrix

    labelGroup = getLabelGroup(data, label, numCluster)
    print labelGroup
    cluster = []
    for i in range(numCluster):
        cluster.append(clusterProcessFunc( data[labelGroup[i],:] ))

    for i in range(numCluster):
        for j in range(numCluster):
            if i < j:
                distMatrix[i][j] = distFunc(cluster[i], cluster[j])
                distMatrix[j][i] = distMatrix[i][j]
            elif i==j:
                distMatrix[i][j] = 0.0
    #print 'distMatrix full:\n', distMatrix
    #normalize distMatrix
    return distMatrix / distMatrix.max()

######################################################
def normalizeDistMatrix(distMatrix):
    distMatrix = np.matrix(distMatrix)
    maxVal = distMatrix.max()
    return (distMatrix / maxVal).tolist()

######################################################
def centerDistFunc(X1, X2):
    return np.linalg.norm(X1 - X2)

######################################################
def pairwiseDistFunc(X1, X2):
    print X1.shape, X2.shape

    pairDistSum = 0.0
    count = 0
    for i in range(X1.shape[0]):
      for j in range(X2.shape[0]):
        if i==j:
           continue
        pairDistSum = pairDistSum + np.linalg.norm(X1[i,:] - X2[j,:])
        count = count+1

    return pairDistSum/count

#####################################################
def doNothing(X):
    return X

######################################################
def principalComponent(X, dim=2):
    pca = decomposition.PCA(n_components=dim)
    pca.fit(X)
    return pca.components_
######################################################
def subspaceDistFunc(X1, X2):
    dist = findDist(np.matrix(X1), np.matrix(X2), 0, 'chordal')
    return dist

######################################################
def computeFlatClusterAccuracy(label, groundTruthLabel):
    #re-index to match groundTruth
    if(len(label)!=len(groundTruthLabel)):
        return -1; #error
    label = bestMap(groundTruthLabel,label)
    return reduce(lambda x,y:x+y, map(lambda l1,l2: 0 if l1==l2 else 1, label, groundTruthLabel))


######################################################
def clusterLinearSubspace(data, labels, numCluster):
    subspaceList = []
    #compute PCA for each cluster
    labelGroup = getLabelGroup(data, label, numCluster)
    for i in xrange(numCluster):
        print "computing cluster:",i,", clusterSize:",label_len[i]
        #X = data[labelGroup[i],:]
        X = data[:,labelGroup[i]]
        #print "data shape:", data.shape,"   cluster dataset shape:", X.shape
        #outlierValue = detectOutlier(X)
        #outlierValue = computeDistanceToClusterCenter(X)
        X = np.transpose(X)
        pca = decomposition.PCA(n_components=2)
        pca.fit(X)

        subspaceList.append(pca.components_) #[n_components, n_features]
    return subspaceList

######################################################
def flatClustering(X, numCluster, method):
    print "flatClustering"
    labels = []

    if numCluster<=1:
        return labels
    # Compute DBSCAN
    if method == 'DBSCAN':
        #db = DBSCAN(eps=0.3, min_samples=10).fit(X)
        db = DBSCAN(eps=3.3, min_samples=3).fit(X)
        #db = DBSCAN().fit(X)
        #core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        #core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        #print "DBSCAN cluster number:", n_clusters
        #print "Warning '-1' label number:", list(labels).count(-1)

        #for index, l in enumerate(labels.tolist()):
        #    if l==-1:
        #        labels[index] = maxLabel+1

    # Compute KMeans++
    elif method == 'kMeans++':
        kmeans = KMeans(init='k-means++', n_clusters=numCluster, n_init=10)
        kmeans.fit(X)
        labels = kmeans.labels_

    # Compute Hierarchy
    elif method == 'Hierarchy-Ward':
        hierarchy = AgglomerativeClustering(n_clusters=numCluster, linkage='ward')
        hierarchy.fit(X)
        labels = hierarchy.labels_

    elif method == 'Hierarchy-CL':
        hierarchy = AgglomerativeClustering(n_clusters=numCluster, linkage='complete')
        hierarchy.fit(X)
        labels = hierarchy.labels_

    elif method == 'Hierarchy-AL':
        hierarchy = AgglomerativeClustering(n_clusters=numCluster, linkage='average')
        hierarchy.fit(X)
        labels = hierarchy.labels_

    # Compute Spectral Clustering
    elif method == 'Spectral':
        spectral = SpectralClustering(n_clusters=numCluster, n_neighbors=20, eigen_solver='arpack',
                                          #affinity="rbf", )
                                          affinity="nearest_neighbors")
        spectral.fit(X)
        labels = spectral.labels_

    elif method == 'PairDir':
        if X.shape[0]%2 != 0:
            print "Odd number of point can not do pair direction"
        vecData = np.squeeze(np.array([ X[i*2][:]-X[i*2+1][:] for i in range(X.shape[0]/2)]))
        print 'vecData:',vecData.shape
        #distMatrix = pairwise_distances(vecData, metric='cosine')
        #print 'distMatrix', distMatrix.shape
        hierarchy = AgglomerativeClustering(n_clusters=numCluster,
                            linkage='average', affinity='cosine')
        hierarchy.fit(vecData)
        vecLabel = hierarchy.labels_
        labels = np.repeat(vecLabel, 2)
        print "labels:", labels.shape

    return labels

######################################################
def subspaceClustering(data, numCluster, method, groundTruthLabel = np.zeros(shape=(1,1))):
    #currently don't use groud truth label
    [label, affinity] = Cpp_embedd_subspaceClustering(data, numCluster, method, groundTruthLabel)
    #estimate basis
    ####!!!! label->shape:(sample,1)
    label = np.asarray(label)
    subspaceList = basisEstimation(data, np.transpose(label), affinity, numCluster)

    return label, subspaceDistanceMatrix(subspaceList), subspaceList

#####################################################
def subspaceDistanceMatrix(subspaceList):
    numCluster = len(subspaceList)
    #compute distance between subspaces from
    distMatrix = np.zeros(shape=(numCluster,numCluster) )
    for index1, subspace1 in enumerate(subspaceList):
        for index2, subspace2 in enumerate(subspaceList):
            #print "Subspace",subspace.shape
            if index1>index2:
                continue
            distMatrix[index1][index2] = findDist(np.matrix(subspace1), np.matrix(subspace2), 0, 'chordal')
            #distMatrix[index1][index2] = findDist(np.transpose(np.matrix(subspace1)), np.transpose(np.matrix(subspace2)), 0, 'chordal')
            distMatrix[index2][index1] = distMatrix[index1][index2]

    return distMatrix

#####################################################
def getLabelGroup(data, label, numCluster):
    labelGroup = [None]*numCluster
    label = label.flatten()
    for index,l in enumerate(label):
        l = int(l)
        if l != -1:
            if labelGroup[l] == None:
                labelGroup[l] = []
            labelGroup[l].append(index)
    return labelGroup

#####################################################
def normalizeVec(vec):
    v_min = np.min(vec)
    v_max = np.max(vec)
    return np.array([ (x-v_min)/(v_max-v_min) for x in vec])

#####################################################
def computeDistanceToSubspace(X, subspaces):
    #find per cluster affinity and compute per-point average distance
    X = np.transpose(X)
    pca = decomposition.PCA(n_components=2)
    #Return the log-likelihood of each sample
    outlierValues = -pca.fit(X).score_samples(X) #convert to the lower the better
    return normalizeVec(outlierValues)

#####################################################
def detectOutlier(X):
    X = np.transpose(X)
    outlierVec = []
    # fit a Minimum Covariance Determinant (MCD) robust estimator to data
    #robust_cov = MinCovDet().fit(X)
    robust_cov = EmpiricalCovariance().fit(X)
    outlierVec = robust_cov.mahalanobis(X)
    #proj
    return np.sqrt(outlierVec)
'''
    visualization
    X2 = dimReduction(X, outputDim=2)
    print "X->X2 - outlierVec shape:", X.shape, X2.shape, outlierVec.shape
    plt.scatter(X2[:, 0], X2[:, 1], c=outlierVec.flatten(), cmap='gray', label='inliers')
    plt.colorbar()
    plt.xticks(())
    plt.yticks(())
    plt.show()
'''
#####################################################
def clusterCenter(X):
    #assume row in X corresponding to points
    mean = np.mean(X, axis=0)
    #print "mean", mean
    return mean

#####################################################
def computeDistanceToClusterCenter(X):
    #print X.shape
    if X.shape[0] < 10:
        return 0.0
    #assume row in X corresponding to points
    mean = np.mean(X, axis=0)
    #print "mean", mean
    dist = np.array([np.linalg.norm(X[index,:]-mean) for index in xrange(X.shape[0])])
    return normalizeVec(dist);

#####################################################
def computeFlatClusterOutlier(data, label):
    numCluster = clusterNumFromLabel(label)
    outlier = np.zeros(len(label.flatten()))
    if numCluster == 1:
        return outlier

    print "=========== begin computing outliers ===========", label.shape
    labelGroup = getLabelGroup(data, label, numCluster)
    label_len=[len(x) for x in labelGroup]

    print "labelGroup-len:", label_len  ,"total size:", sum(label_len);
    for i in xrange(numCluster):
        #print "computing cluster:",i,", clusterSize:",label_len[i]
        X = data[labelGroup[i],:]
        print '  cluster', i, ' ', X.shape
        #print "data shape:", data.shape,"   cluster dataset shape:", X.shape
        #outlierValue = detectOutlier(X)
        outlierValue = computeDistanceToClusterCenter(X)
        #outlierValue = computeDistanceToSubspace(X, subspaces)
        #print "   min max:", min(outlierValue), max(outlierValue)
        #only works if outlier is numpy array
        outlier[labelGroup[i]] = outlierValue

    #normalize
    #o_min = np.min(outlier)
    #o_max = np.max(outlier)
    #norm_outlier = [ (x-o_min)/(o_max-o_min) for x in outlier]
    #print "min, max values:",o_min, o_max
    print 'outlier: ', outlier
    return outlier


#####################################################
def computePerClusterOutlier(data, label, subspaces, numCluster):
    print "=========== begin computing outliers ===========", label.shape
    labelGroup = getLabelGroup(data, label, numCluster)
    label_len=[len(x) for x in labelGroup]

    outlier = np.empty(len(label.flatten()))
    print "labelGroup-len:", label_len  ,"total size:", sum(label_len);
    for i in xrange(numCluster):
        print "computing cluster:",i,", clusterSize:",label_len[i]
        #X = data[labelGroup[i],:]
        X = data[:,labelGroup[i]]
        #print "data shape:", data.shape,"   cluster dataset shape:", X.shape
        #outlierValue = detectOutlier(X)
        #outlierValue = computeDistanceToClusterCenter(X)
        outlierValue = computeDistanceToSubspace(X, subspaces)
        print "   min max:", min(outlierValue), max(outlierValue)
        #only works if outlier is numpy array
        outlier[labelGroup[i]] = outlierValue

    #normalize
    o_min = np.min(outlier)
    o_max = np.max(outlier)
    norm_outlier = [ (x-o_min)/(o_max-o_min) for x in outlier]
    print "min, max values:",o_min, o_max
    return outlier

#####################################################
def dimReduction(X, outputDim = 20):
    pca = decomposition.PCA(n_components=outputDim)
    #pca = decomposition.PCA(n_components='mle')
    pca.fit(X)
    #print pca.explained_variance_ratio_
    #print sum(pca.explained_variance_ratio_)
    #print 'number of component:'
    #print pca.n_components_
    return pca.transform(X)

#####################################################
def dimReductionWithProjMat(X, outputDim = 20):
    pca = decomposition.PCA(n_components=outputDim)
    pca.fit(X)
    return pca.transform(X), pca.components_

#####################################################
def PCAmatrix(X, outputDim):
    pca = decomposition.PCA(n_components=outputDim)
    pca.fit(X)
    return pca.components_, pca.mean_

#####################################################
def queryTimeVaryingWordEmbeddingDatabase(countCutOff, year, dbName, collectionName):
    MONGODB_HOST = 'localhost'
    MONGODB_PORT = 27017
    #DBS_NAME = 'word2Vec'
    #DBS_NAME = 'Glove'
    #COLLECTION_NAME = '50d'

    client = MongoClient(MONGODB_HOST, MONGODB_PORT)
    collection = client[dbName][collectionName]

    cursor = collection.find({"count": {"$gt": countCutOff}, "years": year})
    print str(cursor.count()) + " records found."

    wordDic = {}
    for i in range(cursor.count()):
        wordDic[ cursor[i]["word"] ] = cursor[i]["wordVec"]
        print "word:", cursor[i]["word"]

    print "write to dictionary"
    return wordDic

#####################################################
def getVectorsFromWordEmbeddingDatabase(dbName, collectionName, wordCount=200000):
    #print words
    MONGODB_HOST = 'localhost'
    MONGODB_PORT = 27017

    print "dbName:", dbName, "  collectionName:", collectionName

    client = None
    collection = None
    try:
       client = MongoClient(MONGODB_HOST, MONGODB_PORT)
       collection = client[dbName][collectionName]
       if collection.count() == 0:
           raise ValueError('collection is empty!')
    except:
       print "No local database, connect to remote source"
       client = MongoClient('mongodb://residue3.sci.utah.edu/')
       collection = client[dbName][collectionName]
       if collection.count() == 0:
           print "Can't access remote collection!"

    wordVecs = []
    #this only work for mongoDB 3.2 or up
    #wordCount = len(words)
    #cursor = list(collection.aggregate({"$sample":{"size":wordCount}}))
    #randomIndex = random.sample(range(1, 80000), len(words))
    cursor = list(collection.find({"index":{"$lt":wordCount}}))
    print "collected 100000 word vecs"

    for index, doc in enumerate(cursor):
        wordVecs.append(doc['wordVec'])

    return wordVecs

#####################################################
def queryWordEmbeddingDatabase(words, dbName, collectionName):
    #print words
    MONGODB_HOST = 'localhost'
    MONGODB_PORT = 27017

    print "dbName:", dbName, "  collectionName:", collectionName

    client = None
    collection = None
    try:
       client = MongoClient(MONGODB_HOST, MONGODB_PORT)
       collection = client[dbName][collectionName]
       if collection.count() == 0:
           raise ValueError('collection is empty!')
    except:
       print "No local database, connect to remote source"
       client = MongoClient('mongodb://residue3.sci.utah.edu/')
       collection = client[dbName][collectionName]

    frequencyRank = [0]*len(words)
    wordVecs = None
    originalWords = words
    cursor = list(collection.find({"word":{"$in": words}}))
    print "cursor length:", len(cursor)
    #create word lookup, in case the
    wordIndexLookup = {}
    for index, doc in enumerate(cursor):
        wordIndexLookup[doc["word"]] = index

    #print "word->index map:", wordIndexLookup
    #store the word that didn't got identified
    missedWord = []
    missedWordIndex = []
    wordFound = {}
    for index, word in enumerate(words):
        #init wordFound
        wordFound[word] = True
        if word in wordIndexLookup:
            #print "wordIndex:",wordIndexLookup[word]
            doc = cursor[wordIndexLookup[word]]
            frequencyRank[index] = doc['index']
            #init wordVecs
            if wordVecs == None:
                #print "wordvec length:", len(cursor[0]["wordVec"])
                wordVecs = np.zeros( shape = (len(doc["wordVec"]), len(words)) )
            #replace column
            wordVecs[:,index] = np.array(doc["wordVec"])
        else:
            missedWord.append(originalWords[index].lower()) #save the lower case words
            missedWordIndex.append(index)

    #query the missed words
    if len(missedWord)>0:
        print 'Query missed words(', len(missedWord),'):', missedWord
        cursor = list(collection.find({"word":{"$in":missedWord}}))
        print 'found:', len(cursor)
        wordIndexLookup.clear()
        for index, doc in enumerate(cursor):
            wordIndexLookup[doc["word"]] = index
            frequencyRank[index] = doc['index']
        for index, word in enumerate(missedWord):
            if word in wordIndexLookup:
               doc = cursor[wordIndexLookup[word]]
               #print "found in database using original word"
               # assume result is found
               wordVecs[:, missedWordIndex[index]] = np.array(doc["wordVec"])
            else:
               # the word can not be found in both upper and lower case
               wordFound[word] = False

    print "query words count:", len(words)
    print "wordVecs shape:", wordVecs.shape
    client.close()
    return wordVecs, frequencyRank, wordFound

#####################################################
################ Time Varying Lookup ################
def queryTimeVaryingMostChangedWords(dbName, baseYear, focusedYear, wordDisplayCount):
    MONGODB_HOST = 'localhost'
    MONGODB_PORT = 27017

    client = MongoClient(MONGODB_HOST, MONGODB_PORT)
    collection = client[dbName]['mostChangedWordsFrom-'+str(baseYear)]

    selectedEntries = collection.find({"rank": { '$lt': wordDisplayCount }, "year": focusedYear})
    #print "selected words size: ", selectedEntries.count()
    selectedWordSet = Set()
    for entry in selectedEntries:
        selectedWordSet.add(entry['word'])
    print "selected words: ", selectedWordSet

    mostChangedWords = {}

    for word in selectedWordSet:
        wordInfoArray = []
        wordInDifferenYearEntries = collection.find({"word": word})
        for wordInYear in wordInDifferenYearEntries:
            wordInfoArray.append( [wordInYear['year'], wordInYear['cosDist']] )
        wordInfoArray.sort(key=lambda array:array[0]) #sort the array by year
        mostChangedWords[word] = wordInfoArray

    #print mostChangedWords
    return mostChangedWords

def normalizeList(listVec):
    norm = np.linalg.norm(np.array(listVec))
    #print (np.array(listVec)/norm).tolist()
    #print ""
    return (np.array(listVec)/norm).tolist()

def buildDistMatrixFromWordVec(wordVecsList):
    #build graph from vector
    #for word in wordVecsList:
    #    print word[0], word[2], word[3]

    wordCount = len(wordVecsList)
    #compute distance between subspaces from
    distMatrix = np.zeros(shape=(wordCount, wordCount) )
    for index1, word1 in enumerate(wordVecsList):
        for index2, word2 in enumerate(wordVecsList):
            #print "Subspace",subspace.shape
            if index1>index2:
                continue

            #distMatrix[index1][index2] = spatial.distance.cosine( normalizeList(word1[1]), normalizeList(word2[1]) )
            distMatrix[index1][index2] = 2.0-spatial.distance.cosine(word1[1], word2[1])
            distMatrix[index2][index1] = distMatrix[index1][index2]

    print 'distMatrix min:', np.amin(distMatrix)
    print 'distMatrix max:', np.amax(distMatrix)
    print distMatrix
    return distMatrix.tolist()

def createDiffDatabase(dbName, newCollectionName, year1, year2):
    MONGODB_HOST = 'localhost'
    MONGODB_PORT = 27017
    COLLECTION_NAME = '200d'
    client = MongoClient(MONGODB_HOST, MONGODB_PORT)
    collection = client[dbName][COLLECTION_NAME]

    wordGroup1 = collection.find({"count": { '$gt': 500 }, "years": year1})
    wordGroup2 = collection.find({"count": { '$gt': 500 }, "years": year2})

    set1 = Set()
    set2 = Set()

    for word in wordGroup1:
        if word['word'].isalpha():
            set1.add(word["word"])
    wordGroup1.rewind()

    for word in wordGroup2:
        if word['word'].isalpha():
            set2.add(word["word"])
    wordGroup2.rewind()

    wordSet = set1.intersection(set2)

    print len(set1), len(set2), len(wordSet)

    outputCollection = client[dbName][newCollectionName]

    #cleanup
    outputCollection.remove({})

    wordDiffVec = {}
    cosDist = {}
    count = {}
    for word in wordGroup1:
      if word["word"] in wordSet:
        wordDiffVec[word["word"]] = word["wordVec"]
        count[word["word"]] = word["count"]

    for word in wordGroup2:
      if word["word"] in wordSet:
        tempVec = wordDiffVec[word["word"]]
        wordDiffVec[word["word"]] = (np.array(tempVec) - np.array(word["wordVec"])).tolist()
        cosDist[word['word']] = spatial.distance.cosine(normalizeList(tempVec), normalizeList(word["wordVec"]))
        count[word["word"]] = (count[word["word"]], word["count"])

    for key, value in wordDiffVec.iteritems():
      record = {}
      norm = np.linalg.norm(np.array(value))
      record['word'] = key
      record['count'] = count[key]
      record['norm'] = norm
      record['cosDist'] = cosDist[key]
      record['diffVec'] = value
      outputCollection.insert_one(record)

    outputCollection.create_index('word')

def queryTimeVaryingWord2VecDifferenceByYear(dbName, year1, year2, cutOffRatio):
    MONGODB_HOST = 'localhost'
    MONGODB_PORT = 27017

    client = MongoClient(MONGODB_HOST, MONGODB_PORT)
    collectionName = 'vecDiff-'+str(year1)+'-'+str(year2)

    #check if the collection exist
    if collectionName not in client[dbName].collection_names():
       createDiffDatabase(dbName, collectionName, year1, year2)

    #read out from database
    selectedWordVec = [ ]
    collection = client[dbName][collectionName]
    cursor = collection.find()
    for entry in cursor:
        selectedWordVec.append( (entry['word'], entry['diffVec'], entry['cosDist'], entry['count']) )

    print "cosDist max:", max([x[2] for x in selectedWordVec])
    print "cosDist max:", min([x[2] for x in selectedWordVec])
    selectedWordVec.sort(key=lambda tup: tup[2], reverse=True)

    return selectedWordVec[ : int(len(selectedWordVec)*cutOffRatio)]

def queryTimeVaryingByYear(year):
    MONGODB_HOST = 'localhost'
    MONGODB_PORT = 27017
    DBS_NAME = 'TimeVaryingWord2Vec'
    COLLECTION_NAME = '200d'

    client = MongoClient(MONGODB_HOST, MONGODB_PORT)
    collection = client[DBS_NAME][COLLECTION_NAME]
    #for years in xrange(1900, 2010):
    #    wordGroup = collection.find({"count": { '$gt': 500 }, "years": years})
    #    print years, wordGroup.count()

    wordGroup = collection.find({"count": { '$gt': 500 }, "years": year})
    print year, wordGroup.count()

def queryTimeVaryingWord2VecSimilarityMatrixByYear(dbName, year1, year2):
    MONGODB_HOST = 'localhost'
    MONGODB_PORT = 27017
    #DBS_NAME = 'word2Vec'
    #DBS_NAME = 'Glove'
    #COLLECTION_NAME = '50d'
    pairDict1 = {}
    pairDict2 = {}

    client = MongoClient(MONGODB_HOST, MONGODB_PORT)
    #read year1
    collection = client[dbName]['year-'+ str(year1)]
    cursor = collection.find()
    print "year1 curser count:", cursor.count()

    for pair in cursor:
        pairName = pair['pair']
        dist = pair['dist']
        pairDict1[pairName] = dist

    #read year2
    collection = client[dbName]['year-'+ str(year2)]
    cursor = collection.find()
    print "year2 curser count:", cursor.count()

    for pair in cursor:
        pairName = pair['pair']
        dist = pair['dist']
        pairDict2[pairName] = dist

    #compute differ
    value2 = 0.0
    pairDiff = {}
    wordSet = {}
    for key, value in pairDict1.items():
        #print key, value
        pairKey = key.split('-')
        if pairKey[0].isalpha() and pairKey[1].isalpha():
           #add word to wordSet
           if pairKey[0] not in wordSet:
              wordSet[pairKey[0]] = 0.0;
           if pairKey[1] not in wordSet:
              wordSet[pairKey[1]] = 0.0;

           if key in pairDict2:
              value2 = pairDict2[key]
           else:
              key2 = pairKey[1]+'-'+pairKey[0]
              value2 = pairDict2[key2]
           #if abs(value-value2) > 0.15:
           #pairDiff[key2] = abs(value - value2)
           wordSet[pairKey[0]] = wordSet[pairKey[0]] + abs(value - value2)
           wordSet[pairKey[1]] = wordSet[pairKey[1]] + abs(value - value2)

    #write dictionary to file
    #pairDiffSorted = sorted(pairDiff.items(), key=operator.itemgetter(1))
    #pairDiffSorted.reverse()
    #with open('/home/shusenl/pairDiff.json', 'w') as f:
       #json.dump(pairDiffSorted, f)

    wordSetSorted = sorted(wordSet.items(), key=operator.itemgetter(1))
    wordSetSorted.reverse()
    #with open('/home/shusenl/wordSetSorted.json', 'w') as f:
    #   json.dump(wordSetSorted, f)
#    # load from file:
#    with open('/path/to/my_file.json', 'r') as f:
#        try:
#           data = json.load(f)
#        # if the file is empty the ValueError will be thrown
#        except ValueError:
#           data = {}



############################################
def computePointMultiSense(dataMat, k):
    #find per point neighbor
    dist = np.zeros(len(dataMat))
    for i in range(len(dataMat)):
      #compute distance
      for j in range(len(dataMat)):
          dist[j] = scipy.spatial.distance.cosine(dataMat[i,:], dataMat[j,:])
      #print "partial sort ..."
      kNNIndex = np.argpartition(dist, k)[:k]
      subMat = dataMatrix[kNNIndex, i]

      #cluster for subMat
      spectral = cluster.SpectralClustering(n_clusters=2,
                                          eigen_solver='arpack',
                                          affinity="nearest_neighbors")

      #for j in kNNIndex:
      #   print allWords[j]['word'],
      #print '\n'
