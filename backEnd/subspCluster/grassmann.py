import numpy as np
from scipy import linalg
from math import *

def findGM(A,B):
    n,k = A.shape
    n,l = B.shape
    A = np.matrix(A)
    B = np.matrix(B)
    
    X, Y = findNearest(A,B)
    M = findMidPoint(A,Y)
    N = findMidPoint(B,X)
    
    return M,N
    
def findMidPoint(A, B):
    n,k = A.shape
    n,l = B.shape
    A = np.matrix(A)
    B = np.matrix(B)

    mat = (np.eye(n,n) - A*np.transpose(A))*B*linalg.inv(np.transpose(A)*B)
    Q, Sigma, U =linalg.svd(mat)
    U = np.transpose(U) # V^T is returned by the SVD function
    Q = Q[:,0:len(Sigma)]
    theta = np.arctan(Sigma)
    t1 = np.cos(0.5*np.diag(theta))
    t2 = np.sin(0.5*np.diag(theta))
    M = A*U*np.matrix(t1) + Q*np.matrix(t2)
        
    return M


def findNearest(A, B):
    k = A.shape[1]
    l = B.shape[1]
    
    A, R = linalg.qr(A,mode="economic")
    B, R = linalg.qr(B,mode="economic")
    A = np.matrix(A)
    B = np.matrix(B)
    
    U, Sigma, V = linalg.svd(np.transpose(A)*B) 
    V = np.transpose(V) # V^T is returned by the SVD function
    tempP = A*U
    tempQ = B*V
    
    P = np.concatenate((tempP, tempQ[:,k:l]),axis=1)
    Q = tempQ[:,0:k]
    
    return P,Q
    
def findDist(A, B, ell, opt = "default"):
    U, Sigma, V = linalg.svd(np.transpose(A)*B)
    theta = np.degrees(np.arccos(Sigma))
    theta[np.isnan(theta)] = 0
    K = len(theta)
 
    theta = np.concatenate((theta,90*np.ones(ell)))
    
    if opt == "default":
        dist = np.sqrt(np.sum(theta**2))
    
    if opt == "bider":
        dist = np.sqrt(1 - np.prod(np.cos(np.radians(theta))**2))
    
    if opt == "chordal":
        dist = np.sqrt(np.sum(np.sin(np.radians(theta))**2))
    
    if opt == "fubini":
        dist = np.degrees(np.arccos(np.sum(np.cos(np.radians(theta)))))
    
    if opt == "martin":
        dist = np.sqrt(np.log(np.prod(1/(np.cos(np.radians(theta))**2))))
    
    if opt == "procrustes":
        dist = 2*np.sqrt(np.sum(np.sin(np.radians(0.5*theta))**2))
    
    if opt == "projection":
        dist = np.sin(np.radians(theta[K-1]))
     
    if opt == "spectral":
        dist = 2*np.sin(np.radians(0.5*theta[K-1]))

    return dist