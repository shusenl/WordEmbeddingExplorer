import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from ctypes import *
from word2VecHelper import *
from sys import platform

# #lib = cdll.LoadLibrary('/home/shusenl/gitRepo/DataExplorerHD/build/bin/libdynProjPython.so')
# global lib
# if platform == "linux" or platform == "linux2":
#   #lib = cdll.LoadLibrary('../../releaseBuild/bin/libdynProjPython.so')
#   lib = cdll.LoadLibrary('/home/shusenl/gitRepo/DataExplorerHD/build/bin/libdynProjPython.so')
# elif platform == "darwin":
#   lib = cdll.LoadLibrary('../../releaseBuild/bin/libdynProjPython.dylib')

#FIX_STEPSIZE: the step size is fixed
#FIX_STEPNUM: the number of step is fixed
def generateDynamicProjPathGGobi(projS, projE):
    Fa = np.matrix(projS)
    Fz = np.matrix(projE)

    #print "Fa", Fa
    #print "Fz", Fz
    print "FaDot:", np.dot(np.transpose(Fa[:,0]), Fa[:,1])
    print "FzDot:", np.dot(np.transpose(Fz[:,0]), Fz[:,1])
    dataDim = Fa.shape[0]
    #print "create new obj with dataDim:", dataDim
    lib.SetBeginAndEndBasis( c_int(dataDim), c_void_p(projS.ctypes.data), c_void_p(projE.ctypes.data))
    #print "inited path computation"
    projPath = []
    #lib.interpolateBasisFunction.restype = ndpointer(dtype=c_double, shape=Fa.shape)
    #basis = lib.interpolateBasisFunction(dynProj, c_double(0.5))
    #print "basis", basis
    stepsize = 149
    for i in range(0,stepsize+1):
        t = i/float(stepsize)
        F = np.zeros(projS.shape, dtype=np.double)
        lib.InterpolateBasisFunction(c_double(t), c_void_p(F.ctypes.data))
        #print "basis:\n", F
        #projPath.append(F.tolist())
        projPath.append(F)
        #print "projPath:",t, np.dot(np.transpose(F[:,0]), F[:,1])

    lib.CleanUp()
    return projPath


def generateDynamicProjPath(projS, projE, mode = 'FIX_STEPNUM'):
    #S-start  E-end
    #Fa, Fz, starting and ending frame
    #Ga, Gz, principal vector: starting and ending frame without in plane rotation
    Fa = np.matrix(projS)
    Fz = np.matrix(projE)

    #check if basis are orthonormal
    print "FaDot:", np.dot(np.transpose(Fa[:,0]), Fa[:,1])
    print "FzDot:", np.dot(np.transpose(Fz[:,0]), Fz[:,1])

    #svd
    U, s, V = np.linalg.svd(np.transpose(Fa)*Fz)
    Va = np.matrix(U)
    Vz = np.matrix(V)
    Ga = Fa*Va
    #Ga = normalizeOrthongoalize(Ga)
    #projDataVis(Fa)
    #projDataVis(Ga*np.transpose(Va))

    Gz = Fz*Vz
    #Gz = normalizeOrthongoalize(Gz)

    #orthonormalize Gz on Ga
    #for i in range(0, 2):
    #   Gz[:,i] = gram_schmidt(Ga[:,i], Gz[:,i])

    #Gz = normalizeOrthongoalize(Gz)

    #print 'Ga:',Ga.shape, Ga
    #print 'Gz:', Gz

    #construct starting proj
    tinc = np.zeros(2)
    ptinc = np.zeros((2,2))
    G = np.zeros(Ga.shape)
    for i in range(0,2):
       tmpd1 = np.cos(tinc[i])
       tmpd2 = np.sin(tinc[i])
       for j in range(0,Gz.shape[0]):
           tmpd = Ga[j,i]*tmpd1 + Gz[j,i]*tmpd2
           G[j,i] = tmpd

    #rotate in space of plane to match Fa basis
    F = G*np.transpose(Va)

    #F = normalizeOrthongoalize(F)
    #projDataVis(F)

    print 's:',s
    tau = np.arccos(s)
    print 'principal angle:', tau
    #euclidean norm of principal angles
    dist_az = np.sqrt(np.sum(np.multiply(tau, tau)))
    print 'dist_az:', dist_az
    print 'tau:', tau

    if(dist_az<0.001):
        return [ Fz ] #the span(Fa) is equal to span(Fz)

    #generate path
    projPath = []
    if mode == 'FIX_STEPSIZE':
        projPath.append(Fz)
    elif mode == 'FIX_STEPNUM':
        stepsize = 50
        for i in range(0,stepsize+1):
            t = i/float(stepsize)
            F = projPath_t(tau*t, Ga, Gz, Va)

            projPath.append(F)
            #print "projPath:", np.dot(np.transpose(F[:,0]), F[:,1])

    #pricipalAngle(Fz, np.matrix(projPath[len(projPath)-1]))
    return projPath


def projPath_t(tau_t, Ga, Gz, Va):
    G = np.matrix(np.zeros(Ga.shape))
    for i in range(Ga.shape[1]): #num of column
        ct = np.cos(tau_t[i])
        st = np.sin(tau_t[i])
        for j in range(Ga.shape[0]):
            G[j,i] = ct*Ga[j,i] + st*Gz[j,i]
    # rotate in space of plane to match Fa basis
    F = G*np.transpose(Va)
    print "F:", np.dot(np.transpose(F[:,0]), F[:,1])
    #normalize and orthnogoalize
    #F = normalizeOrthongoalize(F)
    return F

############################# utility ############################

def normalizeBasis(F):
    projDim = F.shape[1]
    print "projDim:", projDim
    for i in range(0, projDim):
        F[:,i] =  normalize(F[:,i])
    return F

def normalizeOrthongoalize(F):
    projDim = F.shape[1]
    for i in range(0, projDim):
        F[:,i] =  normalize(F[:,i])
    for i in range(0, projDim-1):
        for j in range(i+1, projDim):
            F[:,j] = gram_schmidt(F[:,i], F[:,j])
    return F

def pricipalAngle(Fa, Fz):
    U, s, V = np.linalg.svd(np.transpose(Fa)*Fz)
    Va = np.matrix(U)
    Vz = np.matrix(V)
    Ga = Fa*Va
    Gz = Fz*Vz
    tau = np.arccos(s)
    print 'principal angle', tau

########################### gram schmidt ############################
#https://github.com/whille/mylab/blob/master/algrithm/gram_schmidt.py
#gram_schmidt(double *x1, double *x2, int n)
#{
#  int j;
#  double ip;
#  //shusen added more precision NOTE
#  double tol=0.99999;
#  bool ok = true;
#
#  ip = inner_prod(x1, x2, n);
#
#  if (fabs(ip) < tol) { /*  If the two vectors are not orthogonal already */
#    for (j=0; j<n; j++)
#      x2[j] = x2[j] - ip*x1[j];
#    norm(x2, n);
#  }
#  else if (fabs(ip) > 1.0-tol)
#    ok = false;    /* If the two vectors are close to being equal */
#
#  return(ok);
#}
def normalize(v):
    norm=np.linalg.norm(v)
    if norm==0:
       return v
    return v/norm

def gram_schmidt(col1, col2):
  tol=0.99999
  n=col1.shape[0]
  ip = np.dot(np.transpose(col1), col2);
  print "gram-schimidt-ip", ip

  if abs(ip) < tol:
    for j in range(n):
      col2[j] = col2[j] - ip*col1[j];
    normalize(col2);
  return col2

def proj(u, v):
    # notice: this algrithm assume denominator isn't zero
    return u * np.dot(v,u) / np.dot(u,u)

def GS(V):
    V = 1.0 * V     # to float
    U = np.copy(V)
    for i in xrange(1, V.shape[1]):
        for j in xrange(i):
            U[:,i] -= proj(U[:,j], V[:,i])
    # normalize column
    den=(U**2).sum(axis=0) **0.5
    E = U/den
    # assert np.allclose(E.T, np.linalg.inv(E))
    return E

##################### test ######################
def projDataVis(F):
    #X, color = datasets.samples_generator.make_swiss_roll(n_samples=1000)
    #data = X
    data = datasets.load_iris().data
    #print "Data:", data.shape
    #print data
    #print "proj vis:",F
    projData = np.matrix(data)*np.matrix(F)
    #print "projData:", projData.shape
    #print projData
    #projData = np.matrix(np.transpose(F))*np.transpose(data)
    #projData = np.transpose(projData)
    #plt.scatter(data[:,0], data[:,2])
    #plt.scatter(data[:,0], data[:,1])
    #print data[:,0]
    #print data[:,1]
    #plt.show()
    plt.scatter(np.array(np.transpose(projData[:,0])), np.array(np.transpose(projData[:,1])))
    #print np.transpose(projData[:,0])
    #print np.transpose(projData[:,1])
    plt.show()
    #exit()

def randomRotationMatrix(dim):
    rMat = np.random.randn(dim, dim)
    q, r = np.linalg.qr(rMat)
    t = np.dot(q,np.diag(np.sign(np.diag(r))))
    return t #the pure rotation matrix

# def longVecTest():
#  Fa = np.matrix( [[-0.03914039, 0.04377358],
#  [ 0.12795895,  0.17477107],
#  [-0.22191879, -0.06932123],
#  [ 0.02393762, -0.20772931],
#  [-0.10442457, -0.05534904],
#  [ 0.24425818, -0.23976633],
#  [-0.19338494, -0.05602416],
#  [ 0.04484065,  0.01660507],
#  [-0.25444015, -0.15768367],
#  [ 0.00343546,  0.03650383],
#  [ 0.27142054, -0.17392199],
#  [ 0.04464225, -0.15443436],
#  [-0.01254102, -0.12405898],
#  [ 0.05608957,  0.01293842],
#  [ 0.16521949, -0.10450779],
#  [-0.16670766,  0.07550633],
#  [-0.00327122, -0.03040881],
#  [-0.0916068 ,  0.06857506],
#  [-0.20488868, -0.05987949],
#  [-0.12970657,  0.07211869],
#  [-0.01434329,  0.25574036],
#  [-0.20167372,  0.28580734],
#  [-0.23715637, -0.04564345],
#  [ 0.14941242,  0.09169421],
#  [-0.01693128, -0.19366416],
#  [ 0.1784341 ,  0.1804082 ],
#  [-0.03180305, -0.03205659],
#  [-0.02921079,  0.20476831],
#  [-0.15699104, -0.04275618],
#  [-0.12590918, -0.07844747],
#  [-0.11792191, -0.35382766],
#  [ 0.08917371, -0.08501381],
#  [ 0.08628456, -0.06032936],
#  [ 0.03228494, -0.14804837],
#  [-0.17192275,  0.11429512],
#  [-0.17551364,  0.00752574],
#  [ 0.08416663,  0.21548679],
#  [ 0.1751802,  -0.09776219],
#  [ 0.26672296,  0.2368291 ],
#  [-0.08774692,  0.2905091 ],
#  [ 0.17679684,  0.08221234],
#  [-0.17127081,  0.01439089],
#  [ 0.13861795, -0.20805975],
#  [ 0.00112445,  0.10713748],
#  [-0.18060268, -0.07624372],
#  [ 0.04882697, -0.0980227 ],
#  [ 0.06984842,  0.04640588],
#  [-0.13694335, -0.00186348],
#  [ 0.12877144,  0.05242243],
#  [ 0.00308459,  0.02276115]])
#
#  Fz = np.matrix( [[-0.15266801,-0.1125117 ],
#  [ 0.07626837,-0.09190976],
#  [-0.13294476,-0.13329412],
#  [ 0.07442023, -0.17141585],
#  [-0.08637316, -0.0387423 ],
#  [ 0.02742002,  0.16081468],
#  [-0.24030735, -0.22300879],
#  [ 0.06307268,  0.22680155],
#  [ 0.06591138, -0.08387989],
#  [-0.17146881, -0.05869582],
#  [ 0.09154929, -0.06686932],
#  [ 0.00235628,  0.0300278 ],
#  [-0.02485777, -0.05257947],
#  [-0.0938189 , -0.09653868],
#  [ 0.00964522, -0.08385359],
#  [ 0.13644253, -0.16794841],
#  [ 0.1236799 ,  0.11641403],
#  [-0.11741834, -0.14696785],
#  [-0.3213931 , -0.03890844],
#  [ 0.08154212, -0.00276754],
#  [ 0.14518433, -0.47623571],
#  [ 0.05804525, -0.16081821],
#  [-0.03345142,  0.01287617],
#  [ 0.09937336,  0.0271876 ],
#  [-0.01609289, -0.0173205 ],
#  [-0.27532253,  0.01939725],
#  [ 0.16366264,  0.02052586],
#  [-0.06867301, -0.00677913],
#  [-0.14429674,  0.15409627],
#  [ 0.00817499, -0.17310026],
#  [ 0.51537553, -0.01282217],
#  [ 0.09678216,  0.12892472],
#  [ 0.10809304, -0.0797728 ],
#  [-0.09390096,  0.00195261],
#  [ 0.04629283,  0.11789192],
#  [-0.09913907,  0.16827282],
#  [ 0.09296565, -0.0226336 ],
#  [-0.03256167,  0.23106049],
#  [-0.07142686,  0.22166092],
#  [-0.13429374,  0.03546294],
#  [ 0.01299333, -0.01467259],
#  [-0.06272171,  0.0203969 ],
#  [ 0.15282743, -0.17494532],
#  [-0.14001362,  0.14922697],
#  [-0.02210122, -0.30692992],
#  [ 0.28371156,  0.24145648],
#  [-0.15222337,  0.03077237],
#  [-0.00966264,  0.11413251],
#  [ 0.0681611 , -0.01752755],
#  [-0.09108867, -0.07060083]])
#  print Fa.shape, Fz.shape
#  projPath = generateDynamicProjPathGGobi(Fa, Fz)
#  return projPath

def dynamicProjTest():
    dim = 4
    #dim = 3
    #project the data
    #print  data
    unit = np.matrix(np.zeros( (dim,2), dtype=np.double ))
    unit2 = np.matrix(np.zeros( (dim,2), dtype=np.double ))
    unit3 = np.matrix(np.zeros( (dim,2), dtype=np.double ))
    for i in range(2):
      unit[i,i] = 1
    projDataVis(unit)

    unit2[0,0] = 1
    unit2[2,1] = 1
    print unit2
    #projDataVis(unit2)

    #unit3[0,1] = 1
    #unit3[3,0] = 1
    #print unit3
    #projDataVis(np.matrix(unit3))
    #exit()
    #print "unit:", unit

    #rA = np.matrix(randomRotationMatrix(dim))
    #rB = np.matrix(randomRotationMatrix(dim))

    #X, color = datasets.samples_generator.make_swiss_roll(n_samples=1000)
    #data = X
    data = datasets.load_iris().data
    bB, mean = PCAmatrix(data,2)
    print "unit:", unit.shape
    unit2 = normalizeBasis(np.transpose(bB.astype(np.double)))
    #unit2 = np.transpose(bB.astype(np.double))
    print "unit2:", unit2.shape
    #print unit2

    projDataVis(unit)
    projDataVis(unit2)
    #exit()
    #projPath = generateDynamicProjPathGGobi(unit, unit2)
    projPath = generateDynamicProjPath(unit, unit2)
    #print projPath
    #for i in xrange(0,2):
    for i in xrange(len(projPath)):
      projDataVis(np.matrix(projPath[i]))

    #projDataVis(np.matrix(projPath[len(projPath)-1]))
    projDataVis(unit2)


def test():
    V = np.array([[1.0, 1, 1], [1, 0, 2], [1, 0, 0]]).T
    V2=np.copy(V)
    V2[:,2]=[0,1,0]
    E, E2 = GS(V), GS(V2)
    print E, '\n', E2
    # see E[:,2] and E2[:,2] are parallel

    # QR decomposition
    print np.linalg.qr(V)
    print np.dot(E.T, V)
#######################################################################
#longVecTest()
#dynamicProjTest()
