import numpy as np
import matplotlib.pyplot as plt
import numba

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
def deltaE(board, pos):
    oldE = Energy(board)
    board[pos[0]][pos[1]] *= -1
    newE = Energy(board)
    board[pos[0]][pos[1]] *= -1
    return newE-oldE

@numba.njit
def runSweep(board, N, randx, randy, randflip, sweep):
    for step in range(0,N):
        i = sweep*N + step
        if np.exp(-deltaE(board, (randx[i],randy[i]))/T) > randflip[i]:
            board[randx[i]][randy[i]] *= -1
    board_int = ["1" if board[x][y] == 1 else "0" for x in range(L) for y in range(L)]
    bint =''.join(board_int)
    #print(bint)
    return bint

L = 3
T = 10

#board = [[1 for i in range(L)] for j in range(L)]
board = np.random.choice([1,-1], (L,L))

#print(deltaE(board, (1,1)))

sweeps = 100
N = 20

randx = np.random.randint(0,L, sweeps*N)
randy = np.random.randint(0,L, sweeps*N)
randflip = np.random.rand(sweeps*N)
int_list = []

for sweep in range(0,sweeps):
    board_int = runSweep(board, N, randx, randy, randflip, sweep)
    int_list.append(int(board_int,2))

    #print("Sweep:", i, "E =", Energy(board), ", int:", board_int)

prob_int_list = []
total_prob = 0
for i in range(2**(L**2)):
    bit_int = format(i, "0" + str(L**2) + 'b')
    c = 0
    new_board = np.array([[0 for i in range(L)] for j in range(L)])
    for bit in bit_int:
        new_board[c // L][c % L] = 1 if bit == "1" else -1
        c += 1

    prob = np.exp(-Energy(new_board)/T)
    prob_int_list.append(prob)
    total_prob += prob

prob_int_list /= total_prob

xrange = [i for i in range(2**(L**2))]
plt.hist(int_list, xrange)
print(int_list.count(2**(L**2)-1))
plt.scatter(xrange, prob_int_list*(int_list.count(2**(L**2)-1)/max(prob_int_list)), s=4, c='r')
plt.show()