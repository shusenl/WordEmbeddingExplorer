from estimateRandomProjPDF import *
import matplotlib.pyplot as plt

#sampler = wordEmbeddingSampler('Glove', '50d')

#x,y=pairProjErrorPDF(gaussRandomSampler(100), projFunc, pairsEuclideanAnalogyQualityMeasures, 2, 200)
#print x, y
#plotX = [10*i+2 for i in xrange(0,30)]
plotX=[30, 40, 50, 60, 70]
for count in range(5, 6):
    plotZ, _ = zip(*( pairProjErrorPDF(wordEmbeddingSampler('Glove','300d',300), projFunc, pairsEuclideanAnalogyQualityMeasures, 2, 200, i) for i in plotX))
    plotY, _ = zip(*( pairProjErrorPDF(uniformRandomSampler(300), projFunc, pairsEuclideanAnalogyQualityMeasures, 2, 100, i) for i in plotX))
    #plotY, _ = zip(*( pairProjErrorPDF(uniformRandomSampler(i), projFunc, pairsEuclideanAnalogyQualityMeasures, 2, 100, count*10) for i in plotX))
    #plotY = [pairProjErrorPDF(wordEmbeddingSampler('GaussRandom', '300d'), projFunc, pairsEuclideanAnalogyQualityMeasures, 2, 200, 40) for i in plotX]
    plt.plot(plotX, plotY, color='g', label='wordCount:'+str(count*10))
    plt.plot(plotX, plotZ, color='r', label='wordCount:'+str(count*10))

plt.xlabel('dimension')
plt.ylabel('error')
plt.show()
