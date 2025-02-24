import numpy as np
import matplotlib.pyplot as plt
import numba
import stats

@numba.njit
def Energy(board):
    E = 0
    L = len(board)
    for x in range(L):
        for y in range(L):
            E += board[x][y]*board[(x + 1 + L)% L][(y + L)% L]
            E += board[x][y]*board[(x - 1 + L)% L][(y + L)% L]
            E += board[x][y]*board[(x + L)% L][(y + 1 + L)% L]
            E += board[x][y]*board[(x + L)% L][(y - 1 + L)% L]

    return -E

@numba.njit
def Magnetization(board):
    M = 0
    L = len(board)
    for x in range(L):
        for y in range(L):
            M += board[x][y]

    return (M/(L**2))**2

@numba.njit
def deltaEAlt(board, pos):
    oldE = Energy(board)
    board[pos[0]][pos[1]] *= -1
    newE = Energy(board)
    board[pos[0]][pos[1]] *= -1
    return newE-oldE

@numba.njit
def deltaE(board, pos):
    (x,y) = pos
    dE = 0
    dE += board[x][y]*board[(x + 1 + L)% L][(y + L)% L]
    dE += board[x][y]*board[(x - 1 + L)% L][(y + L)% L]
    dE += board[x][y]*board[(x + L)% L][(y + 1 + L)% L]
    dE += board[x][y]*board[(x + L)% L][(y - 1 + L)% L]
    return 4*dE

@numba.njit
def runSweep(board, N, randx, randy, randflip, sweep):
    for step in range(0,N):
        i = sweep*N + step
        if np.exp(-B*deltaE(board, (randx[i],randy[i]))) > randflip[i]:
            board[randx[i]][randy[i]] *= -1
    board_int = ["1" if board[x][y] == 1 else "0" for x in range(L) for y in range(L)]
    bint =''.join(board_int)
    #print(bint)
    return bint

def runSimulation(L, B):
    #board = np.array([[1 for i in range(L)] for j in range(L)])
    board = np.random.choice([1,-1], (L,L))

    sweeps = 1000
    N = 20

    # Generate sweeps*N random numbers (x, y), and whether to flip or not
    randx = np.random.randint(0,L, sweeps*N)
    randy = np.random.randint(0,L, sweeps*N)
    randflip = np.random.rand(sweeps*N)
    int_list = []

    # Run through each sweep
    for sweep in range(0,sweeps):
        # Determine the integer representative of the configuration after doing the sweep (binary string)
        board_int = runSweep(board, N, randx, randy, randflip, sweep)
        # Add that to the list
        int_list.append(int(board_int,2)) # Binary -> Integer
    
    return int_list

def getEnergies(L):
    energy_list = []
    for i in range(2**(L**2)):
        # Get Binary Representation
        bit_int = format(i, "0" + str(L**2) + 'b')
        
        # Create board based on binary
        new_board = np.array([[0 for i in range(L)] for j in range(L)]) # Empty Board

        # Loop through each bit in binary
        c = 0
        for bit in bit_int:
            new_board[c // L][c % L] = 1 if bit == "1" else -1 # Set to either -1 or 1 depending on bit value
            c += 1

        energy_list.append(Energy(new_board))
    
    return energy_list

def getMagnetizations(L):
    magnetization_list = []
    for i in range(2**(L**2)):
        # Get Binary Representation
        bit_int = format(i, "0" + str(L**2) + 'b')
        
        # Create board based on binary
        new_board = np.array([[0 for i in range(L)] for j in range(L)]) # Empty Board

        # Loop through each bit in binary
        c = 0
        for bit in bit_int:
            new_board[c // L][c % L] = 1 if bit == "1" else -1 # Set to either -1 or 1 depending on bit value
            c += 1

        magnetization_list.append(Magnetization(new_board))
    
    return magnetization_list

def getProbabilities(L, B, E):
    prob_int_list = []
    total_prob = 0

    # Loop through all possible configurations
    for i in range(2**(L**2)):
        prob = np.exp(-B*E[i])
        prob_int_list.append(prob)
        total_prob += prob
    
    return prob_int_list/total_prob

# Find Average Energy
def averageEnergy(L, B, PC, E):
    E_avg = 0
    for i in range(2**(L**2)):
        E_avg += PC[i]*E[i]
    
    return E_avg

# Find Average Magnetization
def averageMagnetization(L, B, PC, M):
    M_avg = 0
    for i in range(2**(L**2)):
        M_avg += PC[i]*M[i]
    
    return M_avg

L = 3

# Theoretical Values
EC = getEnergies(L)
MC = getMagnetizations(L)

B_list = np.linspace(0.1, 1.0, 10)

for B in B_list:
    PC = getProbabilities(L, B, EC)
    E_avg = averageEnergy(L, B, PC, EC)
    M_avg = averageMagnetization(L, B, PC, MC)
    (mean_E, variance_E, error_E, autocorrelation_E) = stats.Stats(PC*EC)
    (mean_M, variance_M, error_M, autocorrelation_M) = stats.Stats(PC*MC)
    print("L:", L, ", B:", np.round(B,2), ", Average Energy:", E_avg, ", Average Magnetization:", M_avg)


# First Part Histogram
""" 
L = 3
B = 0.1

PC_exp = runSimulation(L, B)

EC = getEnergies(L)
PC = getProbabilities(L, B, EC)

xrange = [i for i in range(2**(L**2))]
plt.hist(PC_exp, xrange)
plt.scatter(xrange, PC*(PC_exp.count(2**(L**2)-1)/max(PC)), s=4, c='r') # PC_exp scaled such that its max is the same as the max of PE

plt.show() """