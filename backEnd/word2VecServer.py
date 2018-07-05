from flask import Flask
from flask import render_template,request,jsonify
import json
from bson import json_util
from bson.json_util import dumps
import numpy as np
import sys
from sets import Set
# import matplotlib.pyplot as plt

from word2VecHelper import *
# from wordNet import *
from dynamicProj import *
from os import system

from HDpointsGeometry import *
from estimateRandomProjPDF import *
from mongoStorage import mongoStorage

app = Flask(__name__,template_folder='../frontEnd/templates',static_folder='../frontEnd/static')

#these should no longer be used
words = []
wordVecs = None

####################### flask routes ##########################
@app.route("/")
def index():
    #return "test sucess!"
    return render_template("index.html")

#####################################################
###########  Time Varying Word2Vec related ##########
@app.route('/_lookupTimeVaryingDifferenceVecDistMatrix', methods=['POST', 'GET'])
def lookupTimeVaryingDifferenceVecDistMatrix():
    requestJson = request.get_json()
    year1 = requestJson['year1']
    year2 = requestJson['year2']
    displayRatio = float(requestJson['displayRatio'])
    dbName = requestJson['dbName']

    print year1, year2, displayRatio

    wordVecsList = queryTimeVaryingWord2VecDifferenceByYear(dbName, year1, year2, displayRatio)
    distMatrix = buildDistMatrixFromWordVec(wordVecsList)
    graphDic = {}
    graphDic['distMatrix'] = distMatrix
    graphDic['cosDist'] = [x[2] for x in wordVecsList]
    graphDic['word'] = [x[0] for x in wordVecsList]

    return jsonify(**graphDic)

#####################################################
@app.route('/_lookupTimeVaryingMostChangedWords', methods=['POST', 'GET'])
def lookupTimeVaryingMostChangedWords():
    requestJson = request.get_json()
    focusedYear = requestJson['focusedYear']
    baseYear = requestJson['baseYear']
    wordDisplayCount = float(requestJson['wordDisplayCount'])
    dbName = requestJson['dbName']

    mostChangedWords = queryTimeVaryingMostChangedWords(dbName, baseYear, focusedYear, wordDisplayCount)

    return jsonify(**mostChangedWords)


#####################################################
#this is used for holding the wordVecs in the server
@app.route('/_lookupVector', methods=['POST', 'GET'])
def lookupVector():
    global words
    del words[:]
    requestJson = request.get_json()

    words = requestJson['words']
    dbName = requestJson['dbName']
    collectionName = requestJson['collectionName']

    print words, dbName, collectionName

    #query the words
    global wordVecs
    #wordVecs = queryWord2VecDatabase(words)
    wordVecs = queryWordEmbeddingDatabase(words, dbName, collectionName)
    return jsonify(words_count=len(words))

#####################################################
@app.route('/_fetchWordVectors', methods=['POST','GET'])
def fetchWordVectors():
    requestJson = request.get_json()
    wordsQuery = requestJson['words']
    dbName = requestJson['dbName']
    collectionName = requestJson['collectionName']

    #non-global data
    wordVecsQueryResult, frequency, wordFound = queryWordEmbeddingDatabase(wordsQuery, dbName, collectionName)
    #proj to 20D
    #wordVecsReturn = np.transpose(dimReduction( np.transpose(wordVecsQueryResult), 20))
    wordVecsReturn = wordVecsQueryResult

    #convert to dict
    wordVecDict = {}
    for index, word in enumerate(wordsQuery):
        wordVecDict[word] = wordVecsReturn[:,index].tolist()
    #print frequency
    return jsonify({'wordVec':wordVecDict, 'frequencyRank':frequency, 'wordFound':wordFound})

#####################################################
@app.route('/_fetchMostFrequentWordVectors', methods=['POST', 'GET'])
def fetchMostFrequentWordVectors():
    requestJson = request.get_json()
    count = requestJson['count']
    dbName = requestJson['dbName']
    collectionName = requestJson['collectionName']
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
    cursor = list(collection.find({"index":{"$lt":count}}))
    print "cursor length:", len(cursor)
    words = []
    wordVecs = []
    for index, doc in enumerate(cursor):
       words.append(doc["word"])
       wordVecs.append(doc["wordVec"])

    print "query finished"
    return jsonify({'words':words, 'wordVecs':wordVecs})


#####################################################
@app.route('/_fetchTimeVaryingWordVectorsByYear', methods=['POST', 'GET'])
def fetchTimeVaryingWordVectors():
    requestJson = request.get_json()
    year = int(requestJson['years'])
    countCutOff = int(requestJson['cutOff'])
    dbName = requestJson['dbName']
    collectionName = requestJson['collectionName']
    print year, countCutOff, dbName, collectionName
    return jsonify(**queryTimeVaryingWordEmbeddingDatabase(countCutOff, year, dbName, collectionName))

#####################################################
@app.route('/_queryWordNetBySeedWord', methods=['POST', 'GET'])
def queryWordNetBySeedWord():
    requestJson = request.get_json()
    seedWord = requestJson['seedWord']
    tree, words = generateJSONwordTree(seedWord)
    return jsonify({'tree':tree,'words':words})

#####################################################
@app.route('/_queryWordNetDistance', methods=['POST', 'GET'])
def queryWordNetDistance():
    requestJson = request.get_json()
    wordPairs = requestJson['pairs']
    #print wordPairs
    distance = []
    for pair in wordPairs:
        distance.append(queryWordDistance(pair[0], pair[1]))

    return jsonify({'pairDistance':distance})


#####################################################
@app.route('/_computeLinearSVM', methods=['POST', 'GET'])
def computeLinearSVM():
    dataMat = request.get_json()['matrix']
    label = request.get_json()['label']

    dataMat = np.matrix(dataMat)
    label = np.array(label)
    #print dataMat, dataMat.shape
    distance, classification, directionVec = computeClusterBinearyLinearSeparation(dataMat,label,1.0)
    #distance, classification = computeClusterBinearyLinearSeparation(dataMat,label,10000.0)
    cluster1Data = [dataMat[i] for i in range(len(dataMat)) if label[i] == 0]
    cluster2Data = [dataMat[i] for i in range(len(dataMat)) if label[i] == 1]
    maxClusterDistance1 = computeClusterDiameter(cluster1Data, centerDistFunc);
    maxClusterDistance2 = computeClusterDiameter(cluster2Data, centerDistFunc);

    return jsonify(
           {'distance':distance.tolist(),
            'classification':classification.tolist(),
            'label':label.tolist(),
            'classSize1':maxClusterDistance1,
            'classSize2':maxClusterDistance2,
            'directionVec':directionVec.tolist()
           })

#####################################################
@app.route('/_computePerPointMeasure', methods=['POST', 'GET'])
def computePerPointMeasure():
    dataMat = request.get_json()['matrix']
    print 'Input data shape:', dataMat.shape
    measureType = request.get_json()['measureType']
    k = request.get_json()['neighborSize']
    dataMat = np.matrix(dataMat)
    perPointMeasure = []
    if measureType == 'multiSense':
        # assume the return is list
        perPointMeasure = computePointMultiSense(dataMat, k)

    return jsonify({'perPointMeasure':perPointMeasure})


#####################################################
@app.route('/_computePCA', methods=['POST', 'GET'])
def computePCA():
    dataMat = request.get_json()['matrix']
    dataMat = np.matrix(dataMat)
    #print dataMat, dataMat.shape
    projResult, projMatrix = dimReductionWithProjMat(dataMat,2)
    return jsonify({'projResult':projResult.tolist(), 'projMatrix':projMatrix.tolist()})

#####################################################
@app.route('/_computetSNE', methods=['POST', 'GET'])
def computetSNE():
    dataMat = request.get_json()['matrix']
    dataMat = np.matrix(dataMat)
    print "dataMat for tSNE", dataMat.shape
    projResult = tSNEembedding(dataMat,2)
    print "tSNE result:", projResult.shape
    return jsonify({'projResult':projResult.tolist()})

#####################################################
@app.route('/_computeKDE', methods=['POST', 'GET'])
def computeKDE():
    dataMat = request.get_json()['matrix']
    kernelSize = request.get_json()['kernelSize']
    kernelType = request.get_json()['kernelType']
    dataMat = np.matrix(dataMat)
    #print dataMat, dataMat.shape
    density = computeKernelDensityEstimation(dataMat,kernelType,kernelSize)
    return jsonify({'density':density.tolist()})

#####################################################
@app.route('/_computePCAprojMatrix', methods=['POST', 'GET'])
def computePCAprojMatrix():
    #print request.get_json()
    dataMat = request.get_json()['matrix']
    dataMat = np.matrix(dataMat)
    #print dataMat, dataMat.shape
    pc, mean = PCAmatrix(dataMat,2)
    print 'pc:',pc.shape
    return jsonify({'pc':pc.tolist(), 'mean':mean.tolist()})

#####################################################
@app.route('/_computeAnalogyProjMatrix', methods=['POST', 'GET'])
def computeAnalogyProjMatrix():
    dataMat = np.matrix(request.get_json()['matrix'])
    optimizationType = np.matrix(request.get_json()['type'])
    label = request.get_json()['label']
    #print dataMat, dataMat.shape
    #pc, mean = PCAmatrix(dataMat,2)
    #print 'pc:',pc.shape
    projMat, reflection = computeOptimalAnalogyProjection(dataMat, label, optimizationType)
    return jsonify({'projMat':projMat.tolist(), 'reflection':reflection})
    # return jsonify({'projMat':projMat.tolist(), 'reflection':reflection, 'embedding':embedding})

#####################################################
@app.route('/_computeRotationToVector', methods=['POST', 'GET'])
def computeRotationToVector():
    directVec = np.array(request.get_json()['directVec'])
    return jsonify({'reflection':computeRotationAlignToVector(directVec)})

#######################################
@app.route('/_matchGroundTruthLabel', methods=['POST', 'GET'])
def matchGroundTruthLabel():
    gtLabel = request.get_json()['groundTruth']
    label = request.get_json()['label']
    matchResult = {}
    matchResult['accuracy'] = computeFlatClusterAccuracy(label, groundTruthLabel)
    return jsonify(matchResult)

#####################################################
@app.route('/_dynamicProjection', methods=['POST', 'GET'])
def dynamicProjection():
    projS = np.matrix(request.get_json()['startProj']).astype(np.double)
    projE = np.matrix(request.get_json()['endProj']).astype(np.double)
    projMats = generateDynamicProjPathGGobi(projS, projE)
    #projMats = generateDynamicProjPath(projS, projE)
    projMatList = []
    for pMat in projMats:
       projMatList.append(pMat.tolist())
    #projMatList.append(projE.tolist())
    return jsonify({'projMatList':projMatList})

#####################################################
@app.route('/_dimReductionPreProcessing', methods=['POST', 'GET'])
def dimReductionPreProcessing():
    return

#####################################################
@app.route('/_applyCluster', methods=['POST', 'GET'])
def applyCluster():
    #use global variable
    global wordVecs

    #get parameters
    if not request.query_string:
        print "No param is recived!\n"

    #URL encoded parameter need be access through args.get
    #method name
    method = request.args.get('method')
    #subspace distance or cluster distance
    distType = request.args.get('distType')
    #ground truth label
    trueLabel = request.args.get('trueLabel')
    #number of clusters
    numCluster = int(request.args.get('numCluster'))
    print "method:", method
    print "numCluster", numCluster

    methodSelector = {
      "SSC" : 1,
      "LRR" : 2,
      "L2"  : 3,
      "kMeans++" : -1, #not subspace clustering
      "DBSCAN"   : -1, #not subspace clustering
      "Spectral" :-1, #not subspace clustering
      "ground truth label" : -1 #not subspace clustering
    }

    subspaceMethodIndex = methodSelector[method]

    results = {}
    label = []
    distanceMatrix = []
    subspaces = []
    outlier = []

     ######################### subspace cluster
    if subspaceMethodIndex > 0:
        #apply dimension reduction
        #outputDim = 20
        outputDim = 5
        #the pca code require column order, we have row order
        #wordVecsDR = np.transpose(dimReduction(np.transpose(wordVecs), outputDim))
        wordVecsDR = wordVecs
        #np.savetxt('analogyWords.csv', wordVecsDR.T, delimiter=',')
        print "=====wordVecDR====", wordVecsDR.shape
        print wordVecsDR
        if np.isnan(np.sum(wordVecsDR)):
          errorMsg = "wordVecs contain NaN"
          print errorMsg
          return jsonify(result=errorMsg)
        print "=====wordVec====", " (no NaN found)"

        #apply cluster
        npLabel, npDistanceMatrix, npSubspaces = subspaceClustering(wordVecsDR, numCluster, subspaceMethodIndex)
        npOutlier = computePerClusterOutlier(wordVecsDR, npLabel, npSubspaces, numCluster)
        #convert
        label = npLabel.flatten().tolist()
        distanceMatrix = npDistanceMatrix.tolist()
        outlier = npOutlier.flatten().tolist()
        #test
        #label = [1,2,3]
        #distanceMatrix = [[2,3],[4,5]]
    ######################## other flat clustering methods
    else:
        #outputDim = 20
        #X = dimReduction(np.transpose(wordVecs), outputDim)
        X = np.transpose(wordVecs)

        print 'Shape of X - data: ', X.shape
        if method == "ground truth label":
            label = trueLabel
            #### compute the distance matrix #####
            npDistanceMatrix = computeClusterDistMatrix(X, npLabel, centerDistFunc)
            #convert to non numpy array
            distanceMatrix = npDistanceMatrix.tolist()
        else:
            npLabel = flatClustering(X, numCluster, method)
            #### compute the distance matrix #####
            #npDistanceMatrix = computeClusterDistMatrix(X, npLabel, principalComponent, subspaceDistFunc)
            #npDistanceMatrix = computeClusterDistMatrix(X, npLabel, clusterCenter, centerDistFunc)
            npDistanceMatrix = computeClusterDistMatrix(X, npLabel, doNothing, pairwiseDistFunc)

            npOutlier = computeFlatClusterOutlier(X, npLabel)
            #convert to non numpy array
            label = npLabel.flatten().tolist()
            distanceMatrix = npDistanceMatrix.tolist()
            outlier = npOutlier.flatten().tolist()


    results['label'] = label
    results['distanceMatrix'] = normalizeDistMatrix(distanceMatrix)
    #results['outlierValue'] = outlier.flatten().tolist()
    results['outlierValue'] = outlier     #convert and send to client

    return jsonify(**results)

#####################################################
@app.route('/_applySubspaceCluster', methods=['POST', 'GET'])
def applySubspaceCluster():

    wordVecs = request.get_json()['wordVecs']
    wordVecs = np.matrix(wordVecs)
    print wordVecs.shape

    results = {}
    label = []
    distanceMatrix = []
    subspaces = []
    subspacesBasisList = []

    #clustering method
    method = request.get_json()['method']
    print "method:", method
    #number of clusters
    numCluster = int(request.get_json()['numCluster'])
    print "numCluster", numCluster

    subspaceMethodIndex = -1
    methodSelector = {
      "SSC" : 1,
      "LRR" : 2,
      "L2"  : 3,
      "Subspace-SSC" : 1, #handle different input
      "Subspace-LRR" : 2, #handle different input
      "Subspace-L2"  : 3, #handle different input
      "Hierarchy-Ward" :4,
      "Hierarchy-CL" :5,
      "Hierarchy-AL" :6,
      "kMeans++":7,
      "DBSCAN":8,
      "Spectral":9,
      "PairDir":10
    }

    # enable kmeans++
    #method = "kMeans++"
    subspaceMethodIndex = methodSelector[method]

    ######################### subspace cluster
    if subspaceMethodIndex > 0 and subspaceMethodIndex < 4:
        #apply dimension reduction
        #outputDim = 20
        outputDim = 15
        #the pca code require column order, we have row order
        wordVecsDR = np.transpose(wordVecs) #np.transpose(dimReduction(wordVecs, outputDim))
        #wordVecsDR = np.transpose(dimReduction(wordVecs, outputDim))
        #np.savetxt('analogyWords.csv', wordVecsDR.T, delimiter=',')
        print "=====wordVecDR====", wordVecsDR.shape
        #print wordVecsDR
        if np.isnan(np.sum(wordVecsDR)):
          errorMsg = "wordVecs contain NaN"
          print errorMsg
          return jsonify(result=errorMsg)
        print "=====wordVec====", " (no NaN found)"

        #apply cluster
        npLabel, npDistanceMatrix, npSubspaces = subspaceClustering(wordVecsDR, numCluster, subspaceMethodIndex)
        #npOutlier = computePerClusterOutlier(wordVecsDR, npLabel, npSubspaces, numCluster)
        #convert
        label = npLabel.flatten().tolist()
        distanceMatrix = npDistanceMatrix.tolist()
        for i in range(0, numCluster):
             subspacesBasisList.append(npSubspaces[i].tolist())
        #outlier = npOutlier.flatten().tolist()
        #test
        #label = [1,2,3]
        #distanceMatrix = [[2,3],[4,5]]
    else:
        #X = np.transpose(wordVecs)
        X = wordVecs
        npLabel = flatClustering(X, numCluster, method)
        #### compute the distance matrix #####
        #npDistanceMatrix = computeClusterDistMatrix(X, npLabel, principalComponent, subspaceDistFunc)
        npDistanceMatrix = computeClusterDistMatrix(X, npLabel, clusterCenter, centerDistFunc)
        #convert to non numpy array
        label = npLabel.flatten().tolist()
        print "label:", label
        distanceMatrix = npDistanceMatrix.tolist()
        #subspacesBasisList
        #outlier = npOutlier.flatten().tolist()
    results['label'] = label
    results['distanceMatrix'] = normalizeDistMatrix(distanceMatrix) #results['outlierValue'] = outlier.flatten().tolist()
    results['subspaceBasis'] = subspacesBasisList

    return jsonify(**results)


#####################################################
@app.route('/_queryRandomAnalogyProjectionErrorPDF', methods=['POST', 'GET'])
def queryRandomAnalogyProjectionErrorPDF():
    dataDim      = request.get_json()['dataDim']
    wordCount    = request.get_json()['wordCount']
    projMethod   = request.get_json()['projMethod']

    histo = None
    errorMean = None
    projError = -1

    #first query to see if the PDF sample is in the database
    sampleQuery = {}
    sampleQuery['dataDim'] = dataDim
    sampleQuery['wordCount'] = wordCount
    sampleQuery['projMethod'] = projMethod
    mongoDB = mongoStorage()
    result = mongoDB.query(sampleQuery)
    print "pairCount:", wordCount/2
    if result is None:
       print "Can't find in the database"
       return jsonify({})
       #errorMean, histo = pairProjErrorPDF(uniformRandomSampler(dataDim), projMethod, pairsEuclideanAnalogyQualityMeasures, embeddingDim, sampleCount, wordCount)
       #save to database for later
       #sampleQuery['errorMean']=errorMean
       #sampleQuery['histo']=histo
       #mongoDB.save(sampleQuery)
    else:
       #found the result in database
       histo = result['histo']
       errorMean = result['errorMean']

    #compute the error for current embedding
    embedding2D = np.transpose(np.matrix(request.get_json()['embedding']))
    if embedding2D.shape != (0,0):
       #print embedding2D.shape
       projError = pairsEuclideanAnalogyQualityMeasures(embedding2D);
    return jsonify({'histo': histo, 'error': errorMean, 'projError':projError})


#####################################################
##################### to be finished #################
#####################################################
@app.route('/_computeWordsPCA', methods=['POST', 'GET'])
def computeWordsPCA():
    dataMat = request.get_json()['words']
    dataMat = np.matrix(dataMat)
    print dataMat, dataMat.shape
    projResult = dimReduction(dataMat,2)
    return jsonify({'proj':projResult.tolist()})

#####################################################
@app.route('/_computePartialWordsPCA', methods=['POST', 'GET'])
def computePartialWordsPCA():
    dataMat = request.get_json()['words']
    dataMat = np.matrix(dataMat)
    print dataMat, dataMat.shape
    projResult = dimReduction(dataMat,2)
    return jsonify({'proj':projResult.tolist()})


#####################################################
@app.route('/_uploadCSV', methods=['POST'])
def uploadCSV():
    vecDict = request.get_json()['wordVec']
    fileName = request.get_json()['filename']
    self.MONGODB_HOST = 'localhost'
    self.MONGODB_PORT = 27017
    self.DBS_NAME = 'uploadWordVec'
    self.COLLECTION_NAME = fileName
    client = MongoClient(self.MONGODB_HOST, self.MONGODB_PORT)
    # client = MongoClient('mongodb://residue3.sci.utah.edu/')
    collection = client[self.DBS_NAME][self.COLLECTION_NAME]
    #save CSV to database
    collection.insert(vecDict)
    return 0

#####################################################
### temp ############################################
def top10000_KNN(wordsVec, target):
    #lshf = LSHForest()
    #lshf.fit(wordsVec)
    #distances, indices = lshf.kneighbors([target], n_neighbors=50)
    #kdt     = KDTree(wordsVec, leaf_size=30, metric='euclidean')
    #indices = kdt.query(target, k=80, return_distance=True)
    nbrs        = NearestNeighbors(n_neighbors=100,algorithm='brute', metric='cosine').fit(wordsVec)
    distance, indices = nbrs.kneighbors(target.reshape(1,-1))
    return indices[0],distance

#####################################################
@app.route('/_knn_word', methods=['POST'])
def knn_word():
    requestJson = request.get_json()

    words = requestJson['words']
    dbName = requestJson['dbName']
    collectionName = requestJson['collectionName']

    wordString = requestJson['word'].lower()
    querywordVecs = requestJson['wordVecs']
    queryWords = wordString.split(',')
    words   = []
    lines   = tuple(open('../frontEnd/static/exampleData/top20000.txt', 'r'))
    for line in lines:
        words.append(line.rstrip())

    wordCollection   = queryWordEmbeddingDatabase(words, 'Glove', '50d')
    wordVecs         = wordCollection[0].T
    indices,dists    = top10000_KNN(wordVecs, wordVecs[words.index(word)])

    knn_words        = []
    knn_words_vec    = []
    knn_words_dis    = []
    for i in indices:
        knn_words.append(words[i])
        knn_words_vec.append(wordVecs[i])
    print len(knn_words_vec)
    knn_words_dis = distance.pdist(knn_words_vec, "cosine")
    return jsonify({'knn_words':knn_words, 'knn_words_dis':knn_words_dis.tolist()})

#####################################################
@app.route('/_semanticAxisSorting', methods=['POST', 'GET'])
def semanticAxisSorting():
    requestJson = request.get_json()
    words = requestJson['words']
    dbName = requestJson['dbName']
    reflection = np.matrix(requestJson['reflectMat'])
    collectionName = requestJson['collectionName']

    words   = []
    lines   = tuple(open('../frontEnd/static/exampleData/top20000.txt', 'r'))
    for line in lines:
        words.append(line.rstrip())

    wordCollection   = queryWordEmbeddingDatabase(words, dbName, collectionName)
    wordVecs         = wordCollection[0].T

    print 'wordVecs', wordVecs.shape

    axisValues = []
#    for wordVec in wordVecs:
#       rotatedVec = reflection*wordVec
#       axisValues.append((rotatedVec)

    return jsonify({'axisValue': axisValues})

if __name__ == "__main__":
    #test query result
    #print queryWord2VecDatabase(['in','out','off'])
    app.run(host='localhost',port=5000,debug=True)
    #app.run()
