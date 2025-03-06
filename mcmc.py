import numpy as np
import matplotlib.pyplot as plt
import numba
import scipy.interpolate
import scipy.optimize
import stats
import scipy

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

    return -E/2

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
    return 2*dE

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
    if B > 0.4:
        board = np.array([[1 for i in range(L)] for j in range(L)])
    else:
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
            index2 = i*L//(div*div) + j//div
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
    
    return new_board """

def coarsenInts(int_list, L, div):
    new_int_list = []
    for int_board in int_list:
        new_int_list.append(coarsenInt(int_board, L, div))
    return new_int_list

# Renormalization Group

# Testing Int Coarsing
""" fig_count = 0

L = 27
int_list = runSimulation(L, 0.1, 100, 100)

for i in range(100):
    if int_list[i] != boardToInt(intToBoard(int_list[i],L),L):
        print("Broke")

for i in range(100):
    if coarsenInt(int_list[0], L, 3) != boardToInt(coarsenBoard(intToBoard(int_list[0], L), L, 3), L//3):
        print("Broke")

plt.figure(fig_count)
fig_count += 1
plt.matshow(intToBoard(int_list[0], L))
plt.title("Snapshot for Not Coarsened")

plt.figure(fig_count)
fig_count += 1
plt.matshow(coarsenBoard(intToBoard(int_list[0], L), L, 3), L//3)
plt.title("Snapshot for Coarsened")

plt.show() """

# Coarse vs Natural

save = False

B_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

L = 81

fig_count = 0

avg_M_list_1 = []
error_M_list_1 = []

sweeps = 1000
N = 2000

for B in B_list:
    big_int_list = runSimulation(L, B, sweeps, N)
    int_list = coarsenInts(big_int_list, L, 3)

    # Magnetization
    MC = getMagnetizationSnap(L//3, int_list)

    # Averages
    (mean_M, variance_M, error_M, autocorrelation_M) = stats.Stats(np.array(MC))

    print("B:", B, "-->", "N:", N, mean_M, variance_M, error_M, autocorrelation_M)

    avg_M_list_1.append(mean_M)
    error_M_list_1.append(error_M)

avg_M_list_2 = []
error_M_list_2 = [] 

L = L//3

for B in B_list:
    int_list = runSimulation(L, B, sweeps, N)

    # Magnetization
    MC = getMagnetizationSnap(L, int_list)

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

""" avg_M_list_2.insert(0, 0.0)
avg_M_list_2.append(1.0)

B_list.insert(0,-1.0)
B_list.append(1.1)

M_func = scipy.interpolate.interp1d(avg_M_list_2, np.array(B_list))
B_lin = np.linspace(0,1,100)

plt.close()

plt.figure(0)
plt.plot(np.array(B_list[1:-1]), M_func(avg_M_list_1))
plt.plot(B_lin, B_lin, c="r")
plt.xlabel("Beta")
plt.ylabel("R(Beta)")
plt.title("Coarsening Effect") """

avg_M_list_2[0] = 0
avg_M_list_2[-1] = 1

M_func = scipy.interpolate.interp1d(avg_M_list_2, np.array(B_list))
B_lin = np.linspace(0,1,100)

R = scipy.interpolate.interp1d(np.array(B_list), M_func(avg_M_list_1))

plt.close()

plt.figure(0)
plt.plot(B_lin, R(B_lin))
plt.plot(B_lin, B_lin, c="r")
plt.xlabel("Beta")
plt.ylabel("R(Beta)")
plt.title("Coarsening Effect")

# Arrows
B_iter = 0.5
for i in range(3):
    plt.arrow(B_iter, R(B_iter), R(B_iter)-B_iter,0)
    B_iter = R(B_iter)
    if B_iter < 0 or B_iter > 1:
        break
    plt.arrow(B_iter, B_iter, 0, R(B_iter)-B_iter)

if save:
    plt.savefig("Images/R_coarsening.png")

fixed_point = scipy.optimize.root(lambda B: R(B) - B, 0.5).x[0]

R_grad = scipy.interpolate.interp1d(B_lin, np.gradient(R(B_lin), B_lin))

slope = R_grad(fixed_point)

v = np.log(3)/np.log(slope)

print(slope)
print(v)

plt.show()

# Coarse Snapshots
""" save = True

L = 81

B_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 100]

fig_count = 0

for B in B_list:
    N = 1000
    int_list = runSimulation(L, B, 1000, N)

    bit_int = format(int_list[-1], "0" + str(L**2) + 'b')
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
""" L = 27
N = 1000

B_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 100]

fig_count = 0

for B in B_list:
    int_list = runSimulation(L, B, 10000, N)

    bit_int = format(int_list[0], "0" + str(L**2) + 'b')
    new_board = np.array([[0 for i in range(L)] for j in range(L)]) # Empty Board
    c = 0
    for bit in bit_int:
        new_board[c // L][c % L] = 1 if bit == "1" else -1 # Set to either -1 or 1 depending on bit value
        c += 1

    # Energy
    EC = getEnergiesSnap(L, int_list)
    short_EC, short_EPC = getProbabilitiesByData(EC)

    short_EC, short_EPC = zip(*sorted(zip(short_EC, short_EPC)))
    short_EC = list(short_EC)
    short_EPC = list(short_EPC)

    #print(short_EC)
    (mean_E, variance_E, error_E, autocorrelation_E) = stats.Stats(np.array(EC))
    

    short_EC.append(2*short_EC[-1]-short_EC[-2])
    plt.figure(fig_count)
    fig_count += 1
    plt.hist(short_EC[:-1],short_EC, weights=short_EPC)
    plt.xlabel("Energy")    
    plt.ylabel("Probability")
    plt.title("Energy Histogram (B=" + str(B) + ")")

    plt.show() """

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

    bit_int = format(int_list[-1], "0" + str(L**2) + 'b')
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
    #print(short_EC)

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

    if len(short_EC) == 1:
        short_EC.append(short_EC[-1]+1)
    else:
        short_EC.append(2*short_EC[-1]-short_EC[-2])
    plt.figure(fig_count)
    fig_count += 1
    plt.hist(short_EC[:-1],short_EC, weights=short_EPC)
    plt.xlabel("Energy")    
    plt.ylabel("Probability")
    plt.title("Energy Histogram, " + str(B))
    plt.axvline(mean_E, c="r")
    if save:
        plt.savefig("Images/energy_hist-"+str(B)+".png")
    plt.close()

    if len(short_MC) == 1:
        short_MC.append(short_MC[-1]+1)
    else:
        short_MC.append(2*short_MC[-1]-short_MC[-2])
    plt.figure(fig_count)
    fig_count += 1
    plt.hist(short_MC[:-1],short_MC, weights=short_MPC)
    plt.xlabel("Magnetization")
    plt.ylabel("Probability")
    plt.title("Magnetization Histogram, " + str(B))
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
#print(d_avg_E_list)

plt.figure(fig_count)
fig_count += 1
plt.plot(1/np.array(B_list), -(np.array(B_list))**2 * d_avg_E_list)
plt.xlabel("Temperature")
plt.ylabel("Heat Capacity")
plt.title("Heat Capacity by Temperature")
if save:
    plt.savefig("Images/heat_capacity.png") """

#plt.show()

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