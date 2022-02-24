import numpy as np
np.set_printoptions(precision=3)
import numba as nb
import time
import matplotlib.pyplot as plt

# In[1]:
@nb.jit(nopython=True)
def alpha_beta(delX):
    alpha = delX[:NJ-1]/(delX[:NJ-1] + delX[1:])
    beta  = 2.0/(delX[:NJ-1] + delX[1:])
    alphaW, alphaE = [np.empty(NJ), np.empty(NJ)]
    betaW, betaE   = [np.empty(NJ), np.empty(NJ)]

    alphaW[0]  = alphaE[-1]    = 0.0
    alphaW[1:] = alphaE[:NJ-1] = alpha

    betaW[0]   = betaE[-1]     = 0.0
    betaW[1:]  = betaE[:NJ-1]  = beta

    return beta, alphaW, alphaE, betaW, betaE



@nb.jit(nopython=True)
def auto_fill(conc, beta, delV, delT):
    '''
        Input: concentration
        Output: rj, fE, fW, dE, dW, smG

        rj, fE, fW, dE, dW - N x N matrices (NJ of each)
        smG                - N x 1 column vector
        these values are used to compute matrices A, B, D, G, X, Y
    '''

    dcdx            = (conc[:,1:] - conc[:,:NJ-1]) * beta
    dcdxW, dcdxE    = [np.zeros((N,NJ)), np.zeros((N,NJ))]
    dcdxW[:,1:]     = dcdx
    dcdxE[:,:NJ-1]  = dcdx

    fE, fW  = [np.zeros((N,N,NJ)), np.zeros((N,N,NJ))]
    dE, dW  = [np.zeros((N,N,NJ)), np.zeros((N,N,NJ))]
    rj      = np.zeros((N, N, NJ))
    smG     = np.zeros((N, NJ))

# **************
    n = 0
    rj[n,n,1:NJ-1] = -1.0/delT*delV[1:NJ-1]
    fE[:,:,1:NJ-1] = 0.0
    fW[:,:,1:NJ-1] = 0.0
    dE[n,n,1:NJ-1] = -Diff
    dW[n,n,1:NJ-1] = -Diff
    smG[n, 1:NJ-1] = -(-Diff*dcdxW[n,1:NJ-1] - (-Diff*dcdxE[n,1:NJ-1]))

    j = 0
    rj[n,n,j] = 1.0
    smG[n, j] = conc_x0 - rj[n,n,j]*conc[n,j]

    j = NJ-1
    rj[n,n,j] = 1.0
    smG[n, j] = conc_xmax - rj[n,n,j]*conc[n,j]

# **************
    n = 1
    rj[n,n,1:NJ-1] = -1.0/delT*delV[1:NJ-1]
    fE[:,:,1:NJ-1] = 0.0
    fW[:,:,1:NJ-1] = 0.0
    dE[n,n,1:NJ-1] = -Diff
    dW[n,n,1:NJ-1] = -Diff
    smG[n, 1:NJ-1] = -(-Diff*dcdxW[n,1:NJ-1] - (-Diff*dcdxE[n,1:NJ-1]))

    j = 0
    rj[n,n,j] = 1.0
    smG[n, j] = conc_x0*2 - rj[n,n,j]*conc[n,j]

    j = NJ-1
    rj[n,n,j] = 1.0
    smG[n, j] = conc_xmax - rj[n,n,j]*conc[n,j]

# **************
    n = 2
    rj[n,n,1:NJ-1] = -1.0/delT*delV[1:NJ-1]
    fE[:,:,1:NJ-1] = 0.0
    fW[:,:,1:NJ-1] = 0.0
    dE[n,n,1:NJ-1] = -Diff
    dW[n,n,1:NJ-1] = -Diff
    smG[n, 1:NJ-1] = -(-Diff*dcdxW[n,1:NJ-1] - (-Diff*dcdxE[n,1:NJ-1]))

    j = 0
    rj[n,n,j] = 1.0
    smG[n, j] = -conc_x0 - rj[n,n,j]*conc[n,j]

    j = NJ-1
    rj[n,n,j] = 1.0
    smG[n, j] = conc_xmax - rj[n,n,j]*conc[n,j]

    return rj, fE, fW, dE, dW, smG



@nb.jit(nopython=True)
def ABDGXY(alphaW, alphaE, betaW, betaE, rj, fE, fW, dE, dW, smG):

    A = (1.0 - alphaW)*fW - betaW*dW
    B = rj + betaW*dW + alphaW*fW - (1.0 - alphaE)*fE + betaE*dE
    D = -alphaE*fE - betaE*dE
    G = smG
    X = np.zeros((N, N))
    Y = np.zeros((N, N))

    return A, B, D, G, X, Y


@nb.jit(nopython=True)
def MATINV(A):
    '''
    A = [B | D]
    returns
    A = [I | B^-1 * D]      # matmul(inv(B), D)

    if A[:,:N] \= eye(N)
        then deter(B) == 0

    unclear still how to handle situations where the determinant of B is 0, i.e. B is not invertible

        BTRY    - largest value in a row
        BNEXT   - second largest value in a row
        BMAX    - ratio of the two largest magnitude values in a row (BNEXT / BTRY)
        ID      - keeps track of available pivot rows

    pivot row    - row with the smallest BMAX value (excluding already used pivot rows)
    pivot column - max magnitude index in pivot row (excluding already used pivot columns)

    '''
    N = A.shape[0]

    Determ = 0.0
    ID = np.zeros(N)

    for NN in range(N):
        BMAX = 1.1

        for I in range(N):                  # I - row index
            if (ID[I] == 0):
                BNEXT = 0.0
                BTRY = 0.0

                for J in range(N):          # J - column index
                    if (ID[J] == 0):
                        if not(abs(A[I,J]) <= BNEXT):       # if (abs(A[I,J]) > BNEXT)):
                            BNEXT = abs(A[I,J])
                            if not(BNEXT <= BTRY):          # if (BNEXT > BTRY):
                                BNEXT = BTRY
                                BTRY = abs(A[I,J])
                                JC = J

                if not(BNEXT >= BMAX*BTRY):                 # if (BNEXT < BMAX*BTRY):
                    BMAX = BNEXT/BTRY                       # if (BNEXT / BTRY < BMAX)
                    IROW = I
                    JCOL = JC

        if (ID[JC] != 0):
            DETERM = 0.0
            return A

        ID[JCOL] = 1


        # else:
        # swap rows(JCOL, IROW)
        if JCOL != IROW:
            A_temp     = A[JCOL,:].copy()
            A[JCOL,:]  = A[IROW,:]
            A[IROW,:]  = A_temp

        # make the leading value 1
        A[JCOL, :] /= A[JCOL, JCOL]

        # subtract pivot row from all rows BELOW / ABOVE pivot row
        for ii in range(N):
            if ii == JCOL:
                continue
            f = A[ii,JCOL]
            A[ii,:] -= f * A[JCOL,:]

    return A







# @nb.jit(nopython=True)
def BAND_0(B, D, G, X):
    '''
        solves the BAND at j = 0
        𝐁cⱼ + 𝐃c_{j+1} + 𝐗c_{j+2} = 𝐆
        𝐄ⱼ = 𝐁^-1 𝐃
        𝛏ⱼ = 𝐁^-1 𝐆
        𝐱 = 𝐁^-1 𝐗
    '''

    B_inv = np.linalg.inv(B)

    E  = -np.dot(B_inv, D)
    xi =  np.dot(B_inv, G)
    x  = -np.dot(B_inv, X)

    return E, xi, x

# @nb.jit(nopython=True)
def BAND_J(A, B, D, G, E_jm1, xi_jm1):
    '''
        solves the BAND at j
        𝐀c_{j-1} + 𝐁cⱼ + 𝐃c_{j+1} = 𝐆
        𝐄ⱼ = -[𝐁 + 𝐀𝐄_{j-1}]^-1 𝐃
        𝛏ⱼ =  [𝐁 + 𝐀𝐄_{j-1}]^-1 [𝐆 - 𝐀𝛏_{j-1}]
    '''

    B_inv = np.linalg.inv(B + np.dot( A, E_jm1 ))

    E  = -np.dot(B_inv, D)
    xi =  np.dot(B_inv, G - np.dot( A, xi_jm1 ))

    return E, xi

def BAND(A, B, D, G, X, Y):
    '''
        Outputs the arrays 𝐄, 𝛏, and 𝐱
        From the block tridiagonal matrices 𝐀, 𝐁, 𝐃, 𝐗, 𝐘 and 𝐆

        BAND at j = 0
        𝐁cⱼ + 𝐃c_{j+1} + 𝐗c_{j+2} = 𝐆
        𝐄ⱼ = 𝐁^-1 𝐃
        𝛏ⱼ = 𝐁^-1 𝐆
        𝐱 = 𝐁^-1 𝐗

        BAND at j = 1
        𝐀c_{j-1} + 𝐁cⱼ + 𝐃c_{j+1} = 𝐆
        𝐄ⱼ = -[𝐁 + 𝐀𝐄_{j-1}]^-1 [𝐀𝐱 + 𝐃]
        𝛏ⱼ =  [𝐁 + 𝐀𝐄_{j-1}]^-1 [𝐆 - 𝐀𝛏_{j-1}]

        BAND at 1 < j < NJ-1
        𝐀c_{j-1} + 𝐁cⱼ + 𝐃c_{j+1} = 𝐆
        𝐄ⱼ = -[𝐁 + 𝐀𝐄_{j-1}]^-1 𝐃
        𝛏ⱼ =  [𝐁 + 𝐀𝐄_{j-1}]^-1 [𝐆 - 𝐀 𝛏_{j-1}]

        BAND at j = NJ-1
        𝐘c_{j-2} + 𝐀c_{j-1} + 𝐁cⱼ = 𝐆
        cⱼ = -[𝐀 + [𝐘𝐄_{j-2}] 𝐄_{j-1} + 𝐁]^-1 [𝐆 - 𝐘𝛏_{j-2} - [𝐀 + 𝐘𝐄_{j-2}] 𝛏_{j-1} ]
        𝐀 <- 𝐀 + 𝐘𝐄_{j-2}
        𝐆 <- 𝐆 - 𝐘𝛏_{j-2}
        cⱼ = -[𝐁 + 𝐀 𝐄_{j-1}]^-1 [𝐆 - 𝐀 𝛏_{j-1} ]
    '''

    E    = np.empty((N,N,NJ))
    xi   = np.empty((N,NJ))

    E[:,:,0], xi[:,0], x = BAND_0(B[:,:,0], D[:,:,0], G[:,0], X)            # j == 0

    D[:,:,1] = D[:,:,1] + np.dot(A[:,:,1], x)                               # j == 1

    for j in range(1,NJ):
        if j == NJ-1:
            G[:,j]   += -np.dot(Y, xi[:, j-2])                                       # C.27 - RHS
            A[:,:,j] +=  np.dot(Y, E[:,:, j-2])


        E_jm1  = E[:,:,j-1]
        xi_jm1 = xi[:,j-1]

        # G[:,j]   = G[:,j]   - np.dot(A[:,:,j], xi_jm1)
        # B[:,:,j] = B[:,:,j] + np.dot(A[:,:,j], E_jm1)

        E[:,:,j], xi[:,j] = BAND_J(A[:,:,j], B[:,:,j], D[:,:,j], G[:,j], E_jm1, xi_jm1)

    return E, xi, x




# ************************************************************************************************************
# @nb.jit(nopython=True)
def BAND_0_NEWMAN(B, D, G, X):
    '''
        solves the BAND at j = 0
        𝐁cⱼ + 𝐃c_{j+1} + 𝐗c_{j+2} = 𝐆
        𝐄ⱼ = 𝐁^-1 𝐃
        𝛏ⱼ = 𝐁^-1 𝐆
        𝐱 = 𝐁^-1 𝐗
    '''

    # B_inv = np.linalg.inv(B)
    BDGX = np.concatenate([B, D, G, X], axis=1)
    BDGX_inv = MATINV(BDGX)

    E  = -BDGX_inv[:,N:2*N]
    xi =  BDGX_inv[:,2*N]
    x  = -BDGX_inv[:,2*N+1:]

    return E, xi, x

# @nb.jit(nopython=True)
def BAND_J_NEWMAN(B, D, G):
    '''
        solves the BAND at j
        𝐀c_{j-1} + 𝐁cⱼ + 𝐃c_{j+1} = 𝐆
        𝐄ⱼ = -[𝐁 + 𝐀𝐄_{j-1}]^-1 𝐃
        𝛏ⱼ =  [𝐁 + 𝐀𝐄_{j-1}]^-1 [𝐆 - 𝐀𝛏_{j-1}]
    '''

    # B_inv = np.linalg.inv(B + np.dot( A, E_jm1 ))
    BDG = np.concatenate([B, D, G], axis=1)
    BDG_inv = MATINV(BDG)

    E  = -BDG_inv[:, N:2*N]
    xi =  BDG_inv[:, 2*N]

    return E, xi


@nb.jit(nopython=True)
def BAND_NEWMAN(A, B, D, G, X, Y):
    '''
        Outputs the arrays 𝐄, 𝛏, and 𝐱
        From the block tridiagonal matrices 𝐀, 𝐁, 𝐃, 𝐗, 𝐘 and 𝐆

        BAND at j = 0
        𝐁cⱼ + 𝐃c_{j+1} + 𝐗c_{j+2} = 𝐆
        𝐄ⱼ = 𝐁^-1 𝐃
        𝛏ⱼ = 𝐁^-1 𝐆
        𝐱 = 𝐁^-1 𝐗

        BAND at j = 1
        𝐀c_{j-1} + 𝐁cⱼ + 𝐃c_{j+1} = 𝐆
        𝐄ⱼ = -[𝐁 + 𝐀𝐄_{j-1}]^-1 [𝐀𝐱 + 𝐃]
        𝛏ⱼ =  [𝐁 + 𝐀𝐄_{j-1}]^-1 [𝐆 - 𝐀𝛏_{j-1}]

        BAND at 1 < j < NJ-1
        𝐀c_{j-1} + 𝐁cⱼ + 𝐃c_{j+1} = 𝐆
        𝐄ⱼ = -[𝐁 + 𝐀𝐄_{j-1}]^-1 𝐃
        𝛏ⱼ =  [𝐁 + 𝐀𝐄_{j-1}]^-1 [𝐆 - 𝐀 𝛏_{j-1}]

        BAND at j = NJ-1
        𝐘c_{j-2} + 𝐀c_{j-1} + 𝐁cⱼ = 𝐆
        cⱼ = -[𝐀 + [𝐘𝐄_{j-2}] 𝐄_{j-1} + 𝐁]^-1 [𝐆 - 𝐘𝛏_{j-2} - [𝐀 + 𝐘𝐄_{j-2}] 𝛏_{j-1} ]
        𝐀 <- 𝐀 + 𝐘𝐄_{j-2}
        𝐆 <- 𝐆 - 𝐘𝛏_{j-2}
        cⱼ = -[𝐁 + 𝐀 𝐄_{j-1}]^-1 [𝐆 - 𝐀 𝛏_{j-1} ]
    '''
    # slicing G, xi with last dimension = None ensures these objects remain as column vectors
    # G[:, 0, None]
    # xi[:, j-2, None]

    BDGX = np.empty((N,3*N+1))
    BDG  = np.empty((N,2*N+1))

    E    = np.empty((N,N,NJ))
    xi   = np.empty((N,NJ))

    BDGX[:,:N]      = B[:,:,0]
    BDGX[:,N:2*N]   = D[:,:,0]
    BDGX[:,2*N]     = G[:,0]
    BDGX[:,2*N+1:]  = X

    BDGX_inv    = MATINV(BDGX)
    E[:,:,0]    = -BDGX_inv[:, N:2*N]
    xi[:,0]     =  BDGX_inv[:, 2*N]
    x           = -BDGX_inv[:, 2*N+1:]
    # E[:,:,0], xi[:,0], x = BAND_0_NEWMAN(B[:,:,0], D[:,:,0], G[:,0,None], X)

    D[:,:,1] = D[:,:,1] + np.dot(A[:,:,1], x)

    for j in range(1,NJ):
        if j == NJ-1:
            G[:,j]      += -np.dot(Y, xi[:, j-2])                                       # C.27 - RHS
            A[:,:,j]    +=  np.dot(Y, E[:,:, j-2])

        E_jm1  = E[:,:,j-1]
        xi_jm1 = xi[:,j-1]

        G[:,j]      = G[:,j]   - np.dot(A[:,:,j], xi_jm1)
        B[:,:,j]    = B[:,:,j] + np.dot(A[:,:,j], E_jm1)

# ************** NEW **************
        BDG[:,:N]       = B[:,:,j]
        BDG[:,N:2*N]    = D[:,:,j]
        BDG[:,-1]       = G[:,j]

        BDG_inv     = MATINV(BDG)
        E[:,:,j]    = -BDG_inv[:, N:2*N]
        xi[:,j]     =  BDG_inv[:, 2*N]
# ************** NEW **************
        # E[:,:,j], xi[:,j] = BAND_J_NEWMAN(B[:,:,j], D[:,:,j], G[:,j,None])

    return E, xi, x


@nb.jit(nopython=True)
def calc_delC(NJ, E, xi, x):
    '''
        Calculates Δc from 𝐄, 𝛏, and 𝐱
    '''
    delC = xi                                                 # C_k(j) = ξ_k(j) - trial values

    for M in range(NJ-2, -1, -1): #NJ-1, 1, -1
        delC[:,M] += np.dot(E[:, :, M], delC[:, M+1])           # C.22

    delC[:,0] += np.dot(x, delC[:,2])                         # C.17

    return delC

@nb.jit(nopython=True)
def unsteady(initial_conditions, number_time_steps, auto_fill, ABDGXY, BAND_NEWMAN, calc_delC):
    c_prev = initial_conditions
    yield c_prev

    for it in range(number_time_steps):
        ################### CAN BE DONE IN PARALLEL ##########################################################
        rj, fE, fW, dE, dW, smG = auto_fill(c_prev, beta, delV, delT)
        A, B, D, G, X, Y        = ABDGXY(alphaW, alphaE, betaW, betaE, rj, fE, fW, dE, dW, smG)
        ######################################################################################################
        E, xi, x                = BAND_NEWMAN(A, B, D, G, X, Y)        # calculate E, ξ, x
        # E, xi, x                = BAND(A, B, D, G, X, Y)        # calculate E, ξ, x
        delC                    = calc_delC(NJ, E, xi, x)       # calculate Δc

        c_prev = c_prev + delC

        yield c_prev


@nb.jit(nopython=True)
def steady_state(initial_guess, max_iterations, tolerance, auto_fill, ABDGXY, BAND_NEWMAN, calc_delC):
    c_prev = initial_guess
    it = 0
    max_change = tolerance+1
    converged_BOOLEAN = False

    while (it < max_iterations) & (max_change > tolerance):
        rj, fE, fW, dE, dW, smG = auto_fill(c_prev, beta, delV, delT)
        A, B, D, G, X, Y        = ABDGXY(alphaW, alphaE, betaW, betaE, rj, fE, fW, dE, dW, smG)
        E, xi, x                = BAND_NEWMAN(A, B, D, G, X, Y)        # calculate E, ξ, x
        delC                    = calc_delC(NJ, E, xi, x)       # calculate Δc

        c_prev = c_prev + delC

        max_change = np.max(np.abs(delC))
        if max_change < tolerance:
            converged_BOOLEAN = True

        it += 1

    return c_prev, converged_BOOLEAN

# In[4]:


start = time.time()

N = 3
NJ = 42
number_time_steps = 1001
xmax = 1.0
dx = xmax / float(NJ-2)

x_mesh = np.empty(NJ)
x_mesh[0] = 0
x_mesh[-1] = xmax

for j in range(1, NJ-1):
    x_mesh[j] = x_mesh[0] + dx*(float(j) - 0.5)

delX = np.zeros(NJ)
delX[1:NJ-1] = xmax/(NJ-2)
delV = delX



Diff = 1.0e-3

delT = 1.0  # seconds

conc_x0 = 1.0
conc_xmax = 0.0


conc = np.ones((N,NJ))
conc[1,:] = 2
conc[2,:] = -1
beta, alphaW, alphaE, betaW, betaE  = alpha_beta(delX)


C = np.array(list(unsteady(conc, number_time_steps, auto_fill, ABDGXY, BAND_NEWMAN, calc_delC)))
print(time.time() - start)


# In[6]:
color = ['k', 'b', 'r', 'g', 'orange']
for c, _ in zip(color, [0, 1, 10, 100, C.shape[0]-1]):#, 100, 1000]):
    if _ == 0:
        plt.plot(x_mesh, C[_, 0,:], '-', color=c, zorder=200)
        plt.plot(x_mesh, C[_, 1,:], '--', color=c, zorder=200)
        plt.plot(x_mesh, C[_, 2,:], ':', color=c, zorder=200)
    else:
        plt.plot(x_mesh, C[_, 0,:], '-', color = c)
        plt.plot(x_mesh, C[_, 1,:], '--', color = c)
        plt.plot(x_mesh, C[_, 2,:], ':', color = c)

# plt.xlim([0.8, 1])
# plt.ylim([0., 1])
# In[7]:
conc = np.ones((N,NJ))
conc[1,:] = 2
conc[2,:] = -1
max_iterations = 10000
tolerance = 1e-5

start = time.time()
C_SS, converged = steady_state(conc, max_iterations, tolerance, auto_fill, ABDGXY, BAND_NEWMAN, calc_delC)
print(time.time() - start)
print(converged)

plt.plot(x_mesh, C[-1, 0,:], '-', color = c)
plt.plot(x_mesh, C[-1, 1,:], '--', color = c)
plt.plot(x_mesh, C[-1, 2,:], ':', color = c)

plt.plot(x_mesh, C_SS[0,:], '-', color = 'k', zorder=-10, linewidth=2)
plt.plot(x_mesh, C_SS[1,:], '--', color = 'k', zorder=-10, linewidth=2)
plt.plot(x_mesh, C_SS[2,:], ':', color = 'k', zorder=-10, linewidth=2)


# In[5]:
