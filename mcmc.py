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
def deltaE(board, pos):
    L = len(board)
    (x,y) = pos
    dE = 0
    dE += board[x][y]*board[(x + 1 + L)% L][(y + L)% L]
    dE += board[x][y]*board[(x - 1 + L)% L][(y + L)% L]
    dE += board[x][y]*board[(x + L)% L][(y + 1 + L)% L]
    dE += board[x][y]*board[(x + L)% L][(y - 1 + L)% L]
    return 4*dE

@numba.njit
def runSweep(board, B, N, randx, randy, randflip, sweep):
    L = len(board)
    for step in range(0,N):
        i = sweep*N + step
        if np.exp(-B*deltaE(board, (randx[i],randy[i]))) > randflip[i]:
            board[randx[i]][randy[i]] *= -1
    board_int = ["1" if board[x][y] == 1 else "0" for x in range(L) for y in range(L)]
    bint =''.join(board_int)
    #print(bint)
    return bint

def runSimulation(L, B, sweeps, N):
    #board = np.array([[1 for i in range(L)] for j in range(L)])
    board = np.random.choice([1,-1], (L,L))

    # Generate sweeps*N random numbers (x, y), and whether to flip or not
    randx = np.random.randint(0,L, sweeps*N)
    randy = np.random.randint(0,L, sweeps*N)
    randflip = np.random.rand(sweeps*N)
    int_list = []

    # Run through each sweep
    for sweep in range(0,sweeps):
        # Determine the integer representative of the configuration after doing the sweep (binary string)
        board_int = runSweep(board, B, N, randx, randy, randflip, sweep)
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

def getEnergiesSnap(L, int_list):
    EC = []
    for num in int_list:
        bit_int = format(num, "0" + str(L**2) + 'b')
        
        # Create board based on binary
        new_board = np.array([[0 for i in range(L)] for j in range(L)]) # Empty Board

        # Loop through each bit in binary
        c = 0
        for bit in bit_int:
            new_board[c // L][c % L] = 1 if bit == "1" else -1 # Set to either -1 or 1 depending on bit value
            c += 1
        
        EC.append(Energy(new_board))
    return EC

def getMagnetizationSnap(L, int_list):
    MC = []
    for num in int_list:
        bit_int = format(num, "0" + str(L**2) + 'b')
        
        # Create board based on binary
        new_board = np.array([[0 for i in range(L)] for j in range(L)]) # Empty Board

        # Loop through each bit in binary
        c = 0
        for bit in bit_int:
            new_board[c // L][c % L] = 1 if bit == "1" else -1 # Set to either -1 or 1 depending on bit value
            c += 1
        
        MC.append(Magnetization(new_board))
    return MC

def getProbabilitiesByData(data):
    short_data = []
    short_prob = []
    for i in range(len(data)):
        if short_data.count(data[i]) == 0:
            short_data.append(data[i])
            short_prob.append(1)
        else:
            short_prob[short_data.index(data[i])] += 1
    
    sum_prob = sum(short_prob)
    return np.array(short_data), np.array(short_prob)/sum_prob

# Find Average Energy
def averageEnergy(PC, E):
    E_avg = 0
    for i in range(len(PC)):
        E_avg += PC[i]*E[i]
    
    return E_avg

# Find Average Magnetization
def averageMagnetization(PC, M):
    M_avg = 0
    for i in range(len(PC)):
        M_avg += PC[i]*M[i]
    
    return M_avg

def intToBoard(num, L):
    bit_int = format(num, "0" + str(L**2) + 'b')
    new_board = np.array([[0 for i in range(L)] for j in range(L)]) # Empty Board
    c = 0
    for bit in bit_int:
        new_board[c // L][c % L] = 1 if bit == "1" else -1 # Set to either -1 or 1 depending on bit value
        c += 1
    
    return new_board

def boardToInt(board, L):
    return int("".join(["1" if board[x][y] == 1 else "0" for x in range(L) for y in range(L)]),2)

def coarsenBoard(board, L, div):
    new_board = np.zeros((L//div,L//div))
    for i in range(0,L,div):
        for j in range(0,L,div):
            total = 0
            for i2 in range(div):
                for j2 in range(div):
                    total += board[i+i2][j+j2]
            new_board[i//div][j//div] = 1 if total > 0 else -1
    
    return new_board

def coarsenInt(int_board, L, div):
    new_board = 0
    for i in range(0,L,div):
        for j in range(0,L,div):
            #index = i*L + j
            index2 = i + j//div
            total = 0
            for i2 in range(div):
                for j2 in range(div):
                    index = (i+i2)*L + j + j2
                    #print(index, index2, (int_board & (1 << index)) >> index)
                    total += (int_board & (1 << index)) >> index
            new_board += (1 << index2) if total > 4 else 0
    
    return new_board
    

""" def coarsenInt(int_board, L, div):
    new_board = 0
    board = intToBoard(int_board, L)
    new_board = coarsenBoard(board, L, div)
    new_int_board = boardToInt(new_board, L//3)
    
    return new_int_board """

def coarsenInts(int_list, L, div):
    new_int_list = []
    for int_board in int_list:
        new_int_list.append(coarsenInt(int_board, L, div))
    return new_int_list

# Renormalization Group

# Testing Int Coarsing
""" fig_count = 0

L = 9
int_list = runSimulation(L, 0.1, 1, 100)

plt.figure(fig_count)
fig_count += 1
plt.matshow(intToBoard(int_list[0], L))
plt.title("Snapshot for Not Coarsened")

print(coarsenInt(int_list[0], L, 3))

plt.figure(fig_count)
fig_count += 1
plt.matshow(intToBoard(coarsenInt(int_list[0], L, 3), L//3))
plt.title("Snapshot for Coarsened")

plt.show() """

# Coarse vs Natural

save = True

B_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 100]

L = 81

fig_count = 0

avg_M_list_1 = []
error_M_list_1 = []

sweeps = 10000
N = 2000

for B in B_list:
    int_list = coarsenInts(runSimulation(L, B, sweeps, N), L, 3)

    # Magnetization
    MC = getMagnetizationSnap(L, int_list)
    short_MC, short_MPC = getProbabilitiesByData(MC)

    short_MC, short_MPC = zip(*sorted(zip(short_MC, short_MPC)))
    short_MC = list(short_MC)
    short_MPC = list(short_MPC)

    # Averages
    (mean_M, variance_M, error_M, autocorrelation_M) = stats.Stats(np.array(MC))

    print("B:", B, "-->", "N:", N, mean_M, variance_M, error_M, autocorrelation_M)

    avg_M_list_1.append(mean_M)
    error_M_list_1.append(error_M)

avg_M_list_2 = []
error_M_list_2 = []

for B in B_list:
    int_list = runSimulation(L//3, B, sweeps, N)

    # Magnetization
    MC = getMagnetizationSnap(L, int_list)
    short_MC, short_MPC = getProbabilitiesByData(MC)

    short_MC, short_MPC = zip(*sorted(zip(short_MC, short_MPC)))
    short_MC = list(short_MC)
    short_MPC = list(short_MPC)

    # Averages
    (mean_M, variance_M, error_M, autocorrelation_M) = stats.Stats(np.array(MC))

    print("B:", B, "-->", "N:", N, mean_M, variance_M, error_M, autocorrelation_M)

    avg_M_list_2.append(mean_M)
    error_M_list_2.append(error_M)

plt.figure(fig_count)
fig_count += 1
plt.errorbar(1/np.array(B_list), avg_M_list_1, yerr=error_M_list_1) 
plt.errorbar(1/np.array(B_list), avg_M_list_2, yerr=error_M_list_2, c="r")
plt.legend(["Coarsened", "Not Coarsened"])
plt.xlabel("Temperature")
plt.ylabel("Magnetization <M^2>")
plt.title("Comparing Average Magnetization of Coarsed Simulation")
if save:
    plt.savefig("Images/average_magnetization_coarse.png")

# Coarse Snapshots
""" save = True

L = 81

B_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 100]

fig_count = 0

for B in B_list:
    N = 2000
    int_list = runSimulation(L, B, 1, N)

    bit_int = format(int_list[0], "0" + str(L**2) + 'b')
    new_board = np.array([[0 for i in range(L)] for j in range(L)]) # Empty Board
    c = 0
    for bit in bit_int:
        new_board[c // L][c % L] = 1 if bit == "1" else -1 # Set to either -1 or 1 depending on bit value
        c += 1
    
    plt.figure(fig_count)
    fig_count += 1
    plt.matshow(new_board)
    plt.title("Snapshot for 81x81 at B=" + str(B))
    if save:
        plt.savefig("Images/snapshot-RG-0-"+str(B)+".png")
    plt.close()

    plt.figure(fig_count)
    fig_count += 1
    plt.matshow(coarsenBoard(new_board, 81, 3))
    plt.title("Snapshot for 27x27 (1-coarse) at B=" + str(B))
    if save:
        plt.savefig("Images/snapshot-RG-1-"+str(B)+".png")
    plt.close()

    plt.figure(fig_count)
    fig_count += 1
    plt.matshow(coarsenBoard(coarsenBoard(new_board, 81, 3), 27, 3))
    plt.title("Snapshot for 9x9 (2-coarse) at B=" + str(B))
    if save:
        plt.savefig("Images/snapshot-RG-2-"+str(B)+".png")
    plt.close() """


# Measuring

# B = 0 against predicted
""" 
L = 27
B = 0

int_list = runSimulation(L, B, 10000, 100)

# Energy
EC = getEnergiesSnap(L, int_list)
short_EC, short_EPC = getProbabilitiesByData(EC)

short_EC, short_EPC = zip(*sorted(zip(short_EC, short_EPC)))
short_EC = list(short_EC)
short_EPC = list(short_EPC)

short_EC.append(2*short_EC[-1]-short_EC[-2])
plt.figure(0)
plt.hist(short_EC[:-1],short_EC, weights=short_EPC)
plt.xlabel("Energy")    
plt.ylabel("Probability")
plt.title("Energy Histogram")

E_list  = []
for i in range(10000):
    board = np.random.choice([1,-1], (L,L))
    E_list.append(Energy(board))

E_list, P_list = getProbabilitiesByData(E_list)
E_list, P_list = zip(*sorted(zip(E_list, P_list)))
E_list = list(E_list)
P_list = list(P_list)

plt.scatter(E_list, P_list, c="r")
plt.xlabel("Energy")    
plt.ylabel("Probability")
plt.title("Energy Histogram")
plt.savefig("Images\energy_histogram_0_overlay.png") """

# List of B's

""" save = True

L = 27

B_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 100]

fig_count = 0
avg_E_list = []
avg_M_list = []
error_E_list = []
error_M_list = []

for B in B_list:
    #N = int((50/(0.5)**4)*B**4+50)
    N = 2000
    int_list = runSimulation(L, B, 10000, N)

    bit_int = format(int_list[0], "0" + str(L**2) + 'b')
    new_board = np.array([[0 for i in range(L)] for j in range(L)]) # Empty Board
    c = 0
    for bit in bit_int:
        new_board[c // L][c % L] = 1 if bit == "1" else -1 # Set to either -1 or 1 depending on bit value
        c += 1
    
    plt.figure(fig_count)
    fig_count += 1
    plt.matshow(new_board)
    plt.title("Snapshot for B=" + str(B))
    if save:
        plt.savefig("Images/snapshot-"+str(B)+".png")
    plt.close()

    # Energy
    EC = getEnergiesSnap(L, int_list)
    short_EC, short_EPC = getProbabilitiesByData(EC)

    short_EC, short_EPC = zip(*sorted(zip(short_EC, short_EPC)))
    short_EC = list(short_EC)
    short_EPC = list(short_EPC)

    # Magnetization
    MC = getMagnetizationSnap(L, int_list)
    short_MC, short_MPC = getProbabilitiesByData(MC)

    short_MC, short_MPC = zip(*sorted(zip(short_MC, short_MPC)))
    short_MC = list(short_MC)
    short_MPC = list(short_MPC)

    # Averages
    (mean_E, variance_E, error_E, autocorrelation_E) = stats.Stats(np.array(EC))
    (mean_M, variance_M, error_M, autocorrelation_M) = stats.Stats(np.array(MC))

    print("B:", B, "-->", "N:", N, mean_E, variance_E, error_E, autocorrelation_E)
    print("B:", B, "-->", "N:", N, mean_M, variance_M, error_M, autocorrelation_M)

    avg_E_list.append(mean_E)
    avg_M_list.append(mean_M)
    error_E_list.append(error_E)
    error_M_list.append(error_M)

    # Histograms

    short_EC.append(2*short_EC[-1]-short_EC[-2])
    plt.figure(fig_count)
    fig_count += 1
    plt.hist(short_EC[:-1],short_EC, weights=short_EPC)
    plt.xlabel("Energy")    
    plt.ylabel("Probability")
    plt.title("Energy Histogram")
    plt.axvline(mean_E, c="r")
    if save:
        plt.savefig("Images/energy_hist-"+str(B)+".png")
    plt.close()

    short_MC.append(2*short_MC[-1]-short_MC[-2])
    plt.figure(fig_count)
    fig_count += 1
    plt.hist(short_MC[:-1],short_MC, weights=short_MPC)
    plt.xlabel("Magnetization")
    plt.ylabel("Probability")
    plt.title("Magnetization Histogram")
    plt.axvline(mean_M, c="r")
    if save:
        plt.savefig("Images/magnetization_hist-"+str(B)+".png")
    plt.close()

plt.figure(fig_count)
fig_count += 1
plt.errorbar(1/np.array(B_list), avg_E_list, yerr=error_E_list) 
plt.xlabel("Temperature")
plt.ylabel("Energy <E>")
plt.title("Average Energy")
if save:
    plt.savefig("Images/average_energy.png")

plt.figure(fig_count)
fig_count += 1
plt.errorbar(1/np.array(B_list), avg_M_list, yerr=error_M_list) 
plt.xlabel("Temperature")
plt.ylabel("Magnetization <M^2>")
plt.title("Average Magnetization")
if save:
    plt.savefig("Images/average_magnetization.png")

d_avg_E_list = np.gradient(avg_E_list, B_list)
print(d_avg_E_list)

plt.figure(fig_count)
fig_count += 1
plt.plot(1/np.array(B_list), -(1/np.array(B_list))**2 * d_avg_E_list)
plt.xlabel("Temperature")
plt.ylabel("Heat Capacity")
plt.title("Heat Capacity by Temperature")
if save:
    plt.savefig("Images/heat_capacity.png")

plt.show() """

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