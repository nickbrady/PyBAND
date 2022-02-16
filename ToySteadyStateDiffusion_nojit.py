import numpy as np

N = 1
NJ = 22
xmax = 1.0
Diff = 1.0e-1

delT = 1.0  # seconds

conc_x0 = 1.0
conc_xmax = 0.0

delX = np.zeros(NJ)
delX[1:NJ-1] = xmax/(NJ-2)
delV = delX

# In[1]:
def alpha_beta(delX):
    alpha = delX[:NJ-1]/(delX[:NJ-1] + delX[1:])
    beta  = 2.0/(delX[:NJ-1] + delX[1:])
    alphaW, alphaE = [np.zeros(NJ), np.zeros(NJ)]
    betaW, betaE   = [np.zeros(NJ), np.zeros(NJ)]
    alphaW[1:], alphaE[:NJ-1] = [alpha, alpha]
    betaW[1:], betaE[:NJ-1]   = [beta, beta]

    return beta, alphaW, alphaE, betaW, betaE

def auto_fill(conc, beta, delV):

    dc = conc[:,1:] - conc[:,:NJ-1]
    dcdx = dc * beta
    dcdxW, dcdxE = [np.zeros((N,NJ)), np.zeros((N,NJ))]
    dcdxW[:,1:]    = dcdx
    dcdxE[:,:NJ-1] = dcdx

    fE, fW = [np.zeros((N,N,NJ)), np.zeros((N,N,NJ))]
    dE, dW = [np.zeros((N,N,NJ)), np.zeros((N,N,NJ))]
    rj = np.zeros((N, N, NJ))
    smG = np.zeros((N, NJ))

    rj[:,:,1:NJ-1] = -1.0/delT*delV[1:NJ-1]
    fE[:,:,1:NJ-1] = 0.0
    fW[:,:,1:NJ-1] = 0.0
    dE[:,:,1:NJ-1] = -Diff
    dW[:,:,1:NJ-1] = -Diff
    x = -Diff*dcdxW[:,:NJ-1] - (-Diff*dcdxE[:,:NJ-1])
    smG[:,1:NJ-1] = -(-Diff*dcdxW[:,1:NJ-1] - (-Diff*dcdxE[:,1:NJ-1]))

    # boundary conditions
    # c = conc_x0 at x = 0
    # c = conc_xmax at x = xmax
    j = 0
    n = 0
    rj[n,n,j] = 1.0
    smG[n,j] = conc_x0 - rj[n,n,j]*conc[n,j]

    j = NJ-1
    n = 0
    rj[n,n,j] = 1.0
    smG[n,j] = conc_xmax - rj[n,n,j]*conc[n,j]

    return rj, fE, fW, dE, dW, smG

def ABDGXY(alphaW, alphaE, betaW, betaE, rj, fE, fW, dE, dW, smG):

    X, Y = [np.zeros((N, N)), np.zeros((N,N))]

    A = (1.0 - alphaW)*fW - betaW*dW
    B = rj + betaW*dW + alphaW*fW - (1.0 - alphaE)*fE + betaE*dE
    D = -alphaE*fE - betaE*dE
    G = smG
    X = 0.0
    Y = 0.0

    return A, B, D, G, X, Y



def BAND_0_jit(B, D, G, X):

  B_inv = np.linalg.inv(B)

  E  = -np.dot(B_inv, D)
  xi =  np.dot(B_inv, G)
  x  = -np.dot(B_inv, X)

  return E, xi, x


def BAND_J_jit(A, B, D, G, E_jm1, xi_jm1):

  B_inv = np.linalg.inv(B + np.dot( A, E_jm1 ))

  E  = -np.dot(B_inv, D)
  xi =  np.dot(B_inv, G - np.dot( A, xi_jm1 ))

  return E, xi


def BAND(A, B, D, G, X, Y):
    '''
        Outputs the arrays E, ξ, and x
        From the block tridiagonal matrices A, B, D, X, Y and G
    '''

    E    = np.ones((N,N,NJ))
    xi   = np.ones((N,NJ))

    E[:,:,0], xi[:,0], x = BAND_0_jit(B[:,:,0], D[:,:,0], G[:,0], X)

    D[:,:,1] = D[:,:,1] + np.dot(A[:,:,1], x)

    for j in range(1,NJ):
        if j == NJ-1:
            G[:,j] += - np.dot(Y, xi[:, j-2])                                       # C.27 - RHS
            A[:,:,j] += np.dot(Y, E[:,:, j-2])

        E_jm1  = E[:,:,j-1]
        xi_jm1 = xi[:,j-1]
        E[:,:,j], xi[:,j] = BAND_J_jit(A[:,:,j], B[:,:,j], D[:,:,j], G[:,j], E_jm1, xi_jm1)

    return E, xi, x


def calc_delC(NJ, E, xi, x):
  delC = xi                                                 # C_k(j) = ξ_k(j) - trial values

  for M in range(NJ-2, -1, -1): #NJ-1, 1, -1
    delC[:,M] += np.dot(E[:, :, M], delC[:, M+1])           # C.22

  delC[:,0] += np.dot(x, delC[:,2])                         # C.17

  return delC

# In[3]:

conc = np.ones((N,NJ))

%timeit beta, alphaW, alphaE, betaW, betaE = alpha_beta(delX)

%timeit rj, fE, fW, dE, dW, smG = auto_fill(conc, beta, delV)

%timeit A, B, D, G, X, Y = ABDGXY(alphaW, alphaE, betaW, betaE, rj, fE, fW, dE, dW, smG)

%timeit E, xi, x = BAND(A, B, D, G, X, Y)   # calculate E, ξ, x

%timeit delC = calc_delC(NJ, E, xi, x)      # calculate Δc

# conc += delC                        # update the concentration profiles

# print(conc)

# In[4]: