import numpy as np
from sklearn.decomposition import PCA
from sklearn import preprocessing
from scipy import linalg
from numpy import matlib
import matplotlib.pyplot as plt
from grassmann import *

class subsp(object):
    
    def __init__(self, data, graph, outsamp, outgraph):
        self.data = data
        self.graph = graph
        self.outsamp = outsamp
        self.outgraph = outgraph
        self.dim = 0
        self.basis = 0
        self.pcadim = 0
        self.pcabasis = 0
        
    def estDim(self):
        #cutoff = 0.45;
        cutoff = 0.20;
        print "########### cutoff:",cutoff, "###########"
        d = self.data.shape[0]
        pMax = np.min([d,10])
        
        # Estimate Basis using PCA
        pca = PCA(n_components=2)
        pca.fit(np.transpose(self.data))
        #latent = pca.explained_variance_
        #self.pcadim = np.max([2,np.sum(latent/latent[0] > 0.5)])
        self.pcadim = 2 #only take 2D subspace
        #comp = pca.components_
        #self.pcabasis = np.transpose(comp[0:self.pcadim,:])
        self.pcabasis = pca.components_
        
        # Estimate basis using the Subspace Clustering Graph
        rerun = 3 #use PCA basis
        #rerun = 2 #use improved basis estimation algorithm
        nS = self.data.shape[1]
        data = np.matrix(self.data)
        gr = np.matrix(self.graph)
        outSamp = np.matrix(self.outsamp)
        
        M = np.transpose(np.eye(nS) - gr)*(np.eye(nS) - gr)
        E,U = linalg.eig(data*M*np.transpose(data))
        E = np.real(E)
        U = np.real(U)
        tInd = np.argsort(-E)
        
        if rerun == 1:
            rp = np.random.permutation(nS)
            nTest = 10
            U = U[:,tInd[0:pMax]]
            inMed = np.zeros((pMax-2+1, 1))
            outMed = np.zeros((pMax-2+1, 1))
            inErr = np.zeros((nTest, 1))
            outErr = np.zeros((nTest, 1))
            numNeigh = 5
            for p in range(2,pMax+1):
                Up = U[:,0:p]
                embData = np.transpose(Up)*data
                outProj = np.transpose(Up)*outSamp
                
                if np.sum(np.abs(embData)) == 0:
                    inMed[p-2] = 1
                    outMed[p-2] = 1
                else:
                    for i in range(0,nTest):
                        relIds = np.concatenate((np.arange(0,rp[i]),np.arange(rp[i]+1,nS)))
                        relData = embData[:,relIds]
                        z = np.matrix(relData - matlib.repmat(embData[:,rp[i]],1,nS-1))
                        C = np.transpose(z)*z 
                        C = C + np.eye(nS-1,nS-1)*1e-6*np.trace(C)
                        temp = np.matrix(linalg.pinv(C))*np.ones((nS-1,1))
                        temp = np.array(temp)
                        temp = temp/linalg.norm(temp,2)
                        ind = np.argsort(-temp,axis=0)
                        temp[ind[numNeigh:len(temp)]] = 0
                        
                        recon = np.array(np.matrix(relData)*temp)
                        samp = np.array(embData[:,rp[i]])                     
                        e1 = np.sum((samp - recon)**2)/p
                        e2 = np.sum((samp + recon)**2)/p
                        
                        inErr[i] = np.min([e1,e2])
                    
                    for i in range(0,nTest):
                        nOS = outProj.shape[1]
                        z = np.matrix(outProj - matlib.repmat(embData[:,rp[i]],1,nOS))
                        C = np.transpose(z)*z 
                        C = C + np.eye(nOS,nOS)*1e-6*np.trace(C)
                        temp = np.matrix(linalg.pinv(C))*np.ones((nOS,1))
                        temp = np.array(temp)
                        temp = temp/linalg.norm(temp,2)
                        ind = np.argsort(-temp,axis=0)
                        temp[ind[numNeigh:len(temp)]] = 0
                        
                        recon = np.array(np.matrix(outProj)*temp)
                        samp = np.array(embData[:,rp[i]])                     
                        e1 = np.sum((samp - recon)**2)/p
                        e2 = np.sum((samp + recon)**2)/p
                        
                        outErr[i] = np.min([e1,e2])
                    
                    inMed[p-2] = np.median(inErr)
                    outMed[p-2] = np.median(outErr)
            plt.figure()         
            plt.plot(range(2,pMax+1),inMed,'b')
            plt.plot(range(2,pMax+1),outMed,'r')        
            if inMed[0] < outMed[0]:
                pOpt = 2
            if pMax > 2:
                for p in range(0,pMax-1):
                    if inMed[p] < outMed[p]:
                        tt = np.sum(inMed[p+1:len(inMed)] > outMed[p+1:len(inMed)])
                        if tt==0 :
                            pOpt = p+2
                            break
                        if p == pMax-2:
                            pOpt = pMax
            self.dim = pOpt
            self.basis = U[:, 0:self.dim]
        elif rerun==2:
            U = U[:,tInd[0:pMax]]
            grDist = np.zeros((pMax-2+1, 1))
            nOS = outSamp.shape[1]
            outgr = self.outgraph
            M = np.transpose(np.eye(nOS) - outgr)*(np.eye(nOS) - outgr)
            F,V = linalg.eig(outSamp*M*np.transpose(outSamp))
            V = np.real(V)
            sInd = np.argsort(-F)
            V = V[:,sInd[0:pMax]]
            
            for p in range(2,pMax+1):
                Up = U[:,0:p]
                embData = np.transpose(Up)*outSamp
                embData = preprocessing.normalize(embData)
                outProj = np.transpose(V)*outSamp
                outProj = preprocessing.normalize(outProj)
                P,Q = findNearest(np.transpose(embData),np.transpose(outProj))
                grDist[p-2] = findDist(np.transpose(embData), Q, 0, 'chordal')
                #P,Q = findNearest(Up,V)
                #grDist[p-2] = findDist(Up, Q, pMax-p, 'chordal')
                
            grDist = grDist/np.max(grDist)
            id = np.argmax(grDist>=cutoff)
            #if rateinc[id] > 10:
            #    pOpt = id + 3
            #else:
            #    pOpt = 2
            pOpt = id + 2    
            self.dim = pOpt
            #fix to 2D !!!!!!!!!!!!!!!!!!!!!! for word embedding application
            self.dim = 2
            self.basis = U[:, 0:self.dim]
        else:
            print "=============== use PCA basis ================="
            self.dim = self.pcadim
            #self.basis = U[:, 0:self.dim]
            self.basis = np.transpose(self.pcabasis)
        
        
