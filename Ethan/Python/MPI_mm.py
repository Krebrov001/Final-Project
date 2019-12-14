import numpy as np 
from mpi4py import MPI 
import sys
import time

comm = MPI.COMM_WORLD
world_size = comm.Get_size()
world_rank = comm.Get_rank()

#Tags
MATRIX_A = 0
MATRIX_B = 5000
RESULT = 10000

if(world_size < 2):
    print("Need more than 1 process")
    quit()

start = 0.0
end = 0.0
elapsed = 0.0
size = sys.argv[1]
N = int(size)
temp = np.floor(N / (world_size - 1))
chunk_size = int(temp)
remainder = N - (chunk_size * (world_size - 1))

if(world_rank == 0):
    start = time.time()

    #Randomly generate matricies
    matrix_a = np.random.rand(N, N)
    matrix_b = np.random.rand(N, N)
    matrix_result = np.zeros(shape = (N, N))

    matrix_a = matrix_a * 100
    matrix_b = matrix_b * 100
    matrix_a = matrix_a - 50
    matrix_b = matrix_b - 50

    print("Matrix_A:")
    print(matrix_a)
    print("Matrix_B")
    print(matrix_b)

    #Send chunks of Matrix A to workers
    for dest in range(1, world_size):
        for i in range(chunk_size):
            rowIndex = (i) + ((dest - 1) * chunk_size)
            sendMe = matrix_a[rowIndex]
            comm.send(sendMe, dest=dest, tag=(MATRIX_A + rowIndex))
        if(remainder != 0 and dest == (world_size - 1)):
            for i in range(remainder):
                rowIndex = i + (dest * chunk_size)
                sendMe = matrix_a[rowIndex]
                comm.send(sendMe, dest=dest, tag=(MATRIX_A + rowIndex))
    
    #Send chunks of Matrix B to workers
    for dest in range(1, world_size):
        for i in range(chunk_size):
            rowIndex = (i) + ((dest - 1) * chunk_size)
            sendMe = matrix_b[rowIndex]
            comm.send(sendMe, dest=dest, tag=(MATRIX_B + rowIndex))
        if(remainder != 0 and dest == (world_size - 1)):
            for i in range(remainder):
                rowIndex = i + (dest * chunk_size)
                sendMe = matrix_b[rowIndex]
                comm.send(sendMe, dest=dest, tag=(MATRIX_B + rowIndex))
    
    #Recieve processed chunks
    for worker in range(1, world_size):
        for i in range(chunk_size):
            rowIndex = i + ((worker - 1) * chunk_size)
            data = comm.recv(source=worker, tag=(RESULT + rowIndex))
            matrix_result[rowIndex] = data
        if(remainder != 0 and worker == (world_size - 1)):
            for i in range(remainder):
                rowIndex = i + (worker * chunk_size)
                data = comm.recv(source=worker, tag=(RESULT + rowIndex))
                matrix_result[rowIndex] = data
    
    print("RESULT:")
    print(matrix_result)
    end = time.time()
    elapsed = end - start
    print(elapsed)
    timeFile = "time_" + str(N) + ".txt"
    f = open(timeFile, "a+")
    f.write("%.20f\n" % elapsed)
    f.close()


else:
    worker_matrix_a = np.zeros(shape = (chunk_size, N))
    worker_matrix_b = np.zeros(shape = (chunk_size, N))
    worker_result = np.zeros(shape = (chunk_size, N))

    #recieve matrix_a chunk
    for i in range(chunk_size):
        rowIndex = i + ((world_rank - 1) * chunk_size)
        worker_matrix_a[i] = comm.recv(source=0, tag=(MATRIX_A + rowIndex))


    #recive matrix_b chunk
    for i in range(chunk_size):
        rowIndex = i + ((world_rank - 1) * chunk_size)
        worker_matrix_b[i] = comm.recv(source=0, tag=(MATRIX_B + rowIndex))

    
    # Send worker_result
    worker_result = worker_matrix_a * worker_matrix_b
    for i in range(chunk_size):
        rowIndex = i + ((world_rank - 1) * chunk_size)
        sendMe = worker_result[i]
        comm.send(sendMe, dest=0, tag=(RESULT + rowIndex))

    #Last worker recieves remainder
    if(remainder != 0 and world_rank == (world_size - 1)):
        remainder_worker_matrix_a = np.zeros(shape = (remainder, N))
        remainder_worker_matrix_b = np.zeros(shape = (remainder, N))
        remainder_worker_result = np.zeros(shape = (remainder, N))

        #Recieve matrix_a remaidner
        for i in range(remainder):
            rowIndex = i + (world_rank * chunk_size)
            remainder_worker_matrix_a[i] = comm.recv(source=0, tag=(MATRIX_A + rowIndex))

        #recieve Matrix_b remainder
        for i in range(remainder):
            rowIndex = i + (world_rank * chunk_size)
            remainder_worker_matrix_b[i] = comm.recv(source=0, tag=(MATRIX_B + rowIndex))

        #multiply remainders
        remainder_worker_result = remainder_worker_matrix_a * remainder_worker_matrix_b

        #send remainders
        for i in range(remainder):
            rowIndex = i + (world_rank * chunk_size)
            sendMe = remainder_worker_result[i]
            comm.send(sendMe, dest=0, tag=(RESULT + rowIndex))