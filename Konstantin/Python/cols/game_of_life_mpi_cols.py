import sys
import os
import time
import errno
import numpy as np
from mpi4py import MPI
# The Python3 MPI API just calls the C MPI API under the hood.
# MPI_Init() is actually called when you import the MPI module from the mpi4py package.
# So line 3 already initializes MPI.
# Initializing the MPI environment twice in the same program is an error.
# Therefore in order to avoid that error, you remember not to explicitly call MPI.Init()
# but rather remember that the import does that for you.


def print_file_format(filename):
    print("ERROR: " + filename + " could not be read successfully",
          "File format:",
          "N G O",
          "NxN grid",
          "",
          "N - the size of the NxN grid of 0s and 1s",
          "G - the number of generations to iterate through",
          "O - the output generation value",
          sep='\n', file=sys.stderr, flush=True
    )
# end def print_file_format()


def read_in_file(comm, filename, num_processors):
    file = None
    grid_size = None
    num_generations = None
    output_generations = None
    
    # Enforce input file naming conventions.
    
    if (not filename.startswith("input")):
        print("ERROR: the file name %s needs to be in the form inputN, where N is an integer." % filename, file=sys.stderr, flush=True)
        print("Example of a valid file name: input2", file=sys.stderr, flush=True)
        comm.Abort()

    index = len("input")
    try:
        # Use string slicing to get everything in the filename string after the "input" prefix.
        # All those characters after the "input" prefix have to be number.
        number = int(filename[index:])
    except ValueError:
        print("ERROR: the file name %s needs to be in the form inputN, where N is an integer." % filename, file=sys.stderr, flush=True)
        print("Example of a valid file name: input2", file=sys.stderr, flush=True)
        comm.Abort()
    
    try:
        file = open(filename, "r")
    except FileNotFoundError as e:
        message = filename + " could not be opened: "
        print("ERROR:", file=sys.stderr, flush=True)
        print(message + os.strerror(e.errno), file=sys.stderr, flush=True)
        comm.Abort()
    
    # Read in the first line of the file and save the contents into a temporary list.
    # default string.split() is on any whitespace
    temp_list = (file.readline()).split()
    if (len(temp_list) != 3):
        print_file_format(filename)
        comm.Abort()
    
    try:
        grid_size = int(temp_list[0])
    except ValueError:
        print_file_format(filename)
        comm.Abort()
    
    try:
        num_generations = int(temp_list[1])
    except ValueError:
        print_file_format(filename)
        comm.Abort()
    
    try:
        output_generations = int(temp_list[2])
    except ValueError:
        print_file_format(filename)
        comm.Abort()
    
    if (grid_size <= 0):
        print("ERROR:", file=sys.stderr, flush=True)
        print("N - the size of the NxN grid must be > 0", file=sys.stderr, flush=True)
        comm.Abort()
    
    if (grid_size % 8 != 0):
        print("ERROR:", file=sys.stderr, flush=True)
        print("N - the size of the NxN grid must be divisible by 8", file=sys.stderr, flush=True)
        comm.Abort()
    
    if (num_generations <= 0):
        print("ERROR:", file=sys.stderr, flush=True)
        print("G - the number of generations to iterate through must be > 0", file=sys.stderr, flush=True)
        comm.Abort()
    
    if (output_generations <= 0):
        print("ERROR:", file=sys.stderr, flush=True)
        print("O - the output generation value must be > 0", file=sys.stderr, flush=True)
        comm.Abort()
    
    if (num_generations % output_generations != 0):
        print("ERROR:", file=sys.stderr, flush=True)
        print("G - the number of generations to iterate through must be divisible by O - the output generation value", file=sys.stderr, flush=True)
        comm.Abort()
    
    # You need to validate that the number of processors your code is being run with evenly divides the N or size of your grid.
    if (grid_size % num_processors != 0):
        print("ERROR:", file=sys.stderr, flush=True)
        print("N - the size of the NxN grid must be divisible by the number of processors %d" % num_processors, file=sys.stderr, flush=True)
        comm.Abort()
    
    # Easily read the file into memory.
    map = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for row in range(grid_size):
        for col in range(grid_size):
            map[row, col] = file.read(1)
        file.read(1)  # consume the '\n'
    
    file.close()
    file = None
    
    # return more than one value
    return map, grid_size, num_generations, output_generations
# end def read_in_file()


def setCell(former_cell, neighbors):
    if (former_cell == 1):
        if (neighbors < 2 or neighbors > 3):
            return 0
        else:
            return 1
    else:
        if (neighbors == 3):
            return 1
        else:
            return 0
# end def setCell()


# def main():
comm_sz = None  # Number of processes
my_rank = None  # My process rank
# In the rank 0, grid is a 2D numpy array (matrix) containing the entire map.
# In the other ranks, it is n.
grid = None
# In all the ranks, including 0, this is the chunk (row or column) that belongs
# to the current process.
chunk = None
# In all the ranks, including 0, this is the chunk (row or column) that belongs
# to the current process.
# chunk and chunk2 alternate between themselves as the current_chunk and the former_chunk.
chunk2 = None
# In rank 0, this is None.
# In all other ranks, this is a copy of the lowest row in the rank above it.
halo_above = None
# In rank comm_sz-1 (the lowest rank), this is None.
# In all other ranks, this is a copy of the highest row in the rank below it.
halo_below = None
row_above = None
row_below = None
# Because the mpirun executable itself generates standard error and standard output,
# I need to create a new file / output stream where the application's output should be written.
output = None
# Similarly, this is the file / output stream where the application's timing information shoudl be written.
timing = None

grid_size = None
num_generations = None
output_generations = None

comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
comm_sz = comm.Get_size()

argc = len(sys.argv)
if (argc != 2):
    # If incorrect arguments supplied, have rank 0 abort all the processes.
    # You always want to make sure that only one rank aborts all the processes.
    # The other processes do MPI_Recv() for a message that never comes, so they are blocked.
    # If I would have put MPI_Abort() inside the other processes as well, one of them could
    # potentially kill rank 0 before it had a chance to print() the message out to the screen.
    # A single MPI_Abort() terminates all the processes.
    if (my_rank == 0):
        print("Usage:\n$ python3 %s <input_file>\n" % sys.argv[0], file=sys.stderr, flush=True)
        comm.Abort()
    else:
        # In Python, the size and the data type of the MPI_Recv() are inferred.
        # The communicator is also the object on which the method is called.
        # default arguments: buf=None, int source=ANY_SOURCE, Status status=None
        # Function returns the data received.
        dummy = comm.recv(tag = 100)

input_filename = sys.argv[1]

if (my_rank == 0):
    # read_in_file() returns four values
    grid, grid_size, num_generations, output_generations = read_in_file(comm, input_filename, comm_sz)
    
    # send the data to all processes.
    for i in range(comm_sz):
        comm.send(grid_size, i, tag=0)  # tag 0 is for grid_size
        comm.send(num_generations, i, tag=1)  # tag 1 is for num_generations
        comm.send(output_generations, i, tag=2)  # tag 2 is for output_generations
else:
    # Recieving the data is used as a synchronization mechanism for the other processes.
    # Rank 0 reads in the file, which takes a lot of time. I do not want my other processes to run off and leave
    # rank 0 behind. By doing the comm.recv(), they are patiently waiting with the open mouths until rank 0 finally
    # finishes reading in the file and sends them their data.
    #
    # I am using a tagging system to distinguish the data. Rank 0 uses MPI_Bsend() to send the data.
    # There is a slight chance that the messages could be recieved in a different order than they are sent.
    # If that is the case, explicitly provide tags to recieve the messages in the same order that they were sent.
    # There is no efficiency loss to this, but it prevents some bugs that might be caused when using the same tag
    # for all your sent and recieved data.
    grid_size = comm.recv(source=0, tag=0)  # tag 0 is for grid_size
    num_generations = comm.recv(source=0, tag=1)  # tag 1 is for num_generations
    output_generations = comm.recv(source=0, tag=2)  # tag 2 is for output_generations
    
# Open up the output file/stream.
if (my_rank == 0):
    # This code creates the output filename based on the input filename.
    # input_filename    output_filename
    # input1            output1
    # input11           output11
    #
    # Use string slicing to get everything in the filename string after the "input" prefix.
    # All those characters after the "input" prefix are a number, that was verified in read_in_file() function.
    index = len("input")
    number = int(input_filename[index:])
    output_filename = "output" + str(number)
    try:
        output = open(output_filename, "w")
    # This catches all possible exceptions excluding system exceptions such as KeyboardInterrupt and SystemExit.
    except Exception as e:
        print("ERROR: Could not open the file %s for writing output.\n" % output_filename, file=sys.stderr, flush=True)
        comm.Abort()
    # Success!
    dummy = 'A';
    # Send the success message to all processes.
    for i in range(comm_sz):
        # obj=dummy, int dest=i, int tag=100
        comm.send(dummy, i, 100)
# Make the other ranks either wait for a success message to be sent, or be killed upon failure.
# This is used as a synchronization device.
else:
    # default arguments: buf=None, Status status=None
    # Function returns the data received.
    dummy = comm.recv(source=0, tag=100)

# Open up the timing file/stream.
if (my_rank == 0):
    # This code creates the timing filename based on the input filename.
    # input_filename    timing_filename
    # input1            timing1
    # input11           timing11
    #
    # Use string slicing to get everything in the filename string after the "input" prefix.
    # All those characters after the "input" prefix are a number, that was verified in read_in_file() function.
    index = len("input")
    number = int(input_filename[index:])
    timing_filename = "timing" + str(number)
    try:
        timing = open(timing_filename, "a")
    # This catches all possible exceptions excluding system exceptions such as KeyboardInterrupt and SystemExit.
    except Exception as e:
        print("ERROR: Could not open the file %s for writing output.\n" % timing_filename, file=sys.stderr, flush=True)
        comm.Abort()
    # Success!
    dummy = 'A';
    # Send the success message to all processes.
    for i in range(comm_sz):
        # obj=dummy, int dest=i, int tag=100
        comm.send(dummy, i, 100)
# Make the other ranks either wait for a success message to be sent, or be killed upon failure.
# This is used as a synchronization device.
else:
    # default arguments: buf=None, Status status=None
    # Function returns the data received.
    dummy = comm.recv(source=0, tag=100)

# These variables are relevant for timing information.
# Each rank gets it's own copy of these variables,
# but only the local ones are relevant to a single rank.
# And elapsed is only relevant to rank 0.
# It contains the total elapsed time for the whole entire MPI program,
# including the running time of all the ranks.
local_start = 0.0
local_finish = 0.0
local_elapsed = 0.0
elapsed = 0.0

# Makes all the ranks pause and wait for each other before continuing down.
comm.barrier()

# Start the timer in each of the ranks.
# This timer, like MPI_Wtime() in C, returns the elapsed wall-clock time.
local_start = time.time()

# Setup the chunks.
chunk_size = grid_size * grid_size // comm_sz  # the number of elements in a chunk
num_rows = chunk_size // grid_size  # the number of rows in a chunk
chunk  = np.zeros((num_rows, grid_size), dtype=np.uint8)
chunk2 = np.zeros((num_rows, grid_size), dtype=np.uint8)

# Setup the halos.
if (my_rank != comm_sz - 1):
    halo_above = np.zeros(grid_size, dtype=np.uint8)

if (my_rank != 0):
    halo_below = np.zeros(grid_size, dtype=np.uint8)

# Setup the temp chunk.
if (my_rank == 0):
    # This temp_chunk is in fact a rotated version of the original chunk.
    # So it's size should be viewed as np.zeros((grid_size, num_rows), dtype=np.uint8)
    temp_chunk = np.zeros((num_rows, grid_size), dtype=np.uint8)

""" This code simulates a column-wise Scatter. """
if (my_rank == 0):
    # rank 0 sends a chunk to all other ranks, including itself.
    # In fact, it first sends the chunk to itself.
    # However, I am using comm.Ssend(), which only returns after the message has been recieved
    # by the other end. If there is no recieve at the other end, there is a deadlock.
    # So I need to call comm.Recv() before calling the comm.Ssend(), but then comm.Recv() only returns
    # after the message has been recieved by itself, and because no one has sent anything yet,
    # it would also be a deadlock. I need to initiate a comm.Irecv() to get rid of the deadlock.
    request = comm.Irecv(buf=chunk, source=0, tag=4)
    
    # This outer loop iterates through all the column-wise chunks in the original grid.
    # For each rank, fill in a temp_chunk, and send it over.
    for i in range(0, grid_size, num_rows):
        # This loop copies a single column-wise chunk in the original grid into the temp_chunk.
        # Each column in a column-wise chunk in the original grid is copied into a row in the temp chunk.
        col = i + num_rows-1
        while (col >= i):
            temp_chunk[num_rows-1-col+i, :] = grid[:, col]
            col -= 1
        
        # Send the temp_chunk to the chunk in the corresponding rank.
        # I am using comm.Ssend() to be sure that a given chunk has been recieved by the destination rank,
        # before overwriting the temp_chunk and sending out the next one.
        # This is a Blocking send in synchronous mode.
        # tag 4 is for messages feigning collective communications.
        comm.Ssend(buf=temp_chunk, dest=i / num_rows, tag=4)
    
    # Complete the comm.Irecv() initiated up above.
    request.Wait()

else:
    # Recieve your chunk from rank 0.
    # default argument: Status status=None
    comm.Recv(buf=chunk, source=0, tag=4)

# Copy the scattered data from the chunk into chunk2.
for row in range(num_rows):
    for col in range(grid_size):
        chunk2[row, col] = chunk[row, col]

current_chunk = None
former_chunk = None

for generation in range(num_generations):
    # In Python, assignments of lists just copy the underlying pointers,
    # achieving the same effect as in C.
    if (generation % 2 == 0):
        current_chunk = chunk2
        former_chunk = chunk
    else:
        current_chunk = chunk
        former_chunk = chunk2

    """ The halo arrays get updated each iteration. """
    
    request1 = None
    request2 = None
    # If you have the below buffer.
    if (halo_below is not None):
        # Send the bottom row in the chunk to the rank directly below you.
        request1 = comm.Issend(buf=former_chunk[num_rows-1], dest=my_rank-1, tag=3)
    
    # If you have the above buffer.
    if (halo_above is not None):
        # Send the top row in the chunk to the rank directly above you.
        request2 = comm.Issend(buf=former_chunk[0], dest=my_rank+1, tag=3)

    # If you have the below buffer.
    if (halo_below is not None):
        # Recieve the top row in the chunk from the rank directly below you.
        comm.Recv(buf=halo_below, source=my_rank-1, tag=3)
    
    # If you have the above buffer.
    if (halo_above is not None):
        # Recieve the bottom row in the chunk from the rank directly above you.
        comm.Recv(buf=halo_above, source=my_rank+1, tag=3)
    
    # If you have the below buffer.
    if (halo_below is not None):
        # default argument Status status=None
        request1.Wait()
    
    # If you have the above buffer.
    if (halo_above is not None):
        # default argument Status status=None
        request2.Wait()

    # Process the data.
    # This loops through all the rows.
    # In a single iteration of this loop, an entire row of cells, with all the columns, is computed.
    # Before performing the actual Game of Life Algorithm, the rows above and below that cell are pre-computed
    # in order to make it easier.
    for row in range(num_rows):
        if (my_rank == comm_sz-1):  # top rank
            # determine what the row above should be
            if (row == 0):
                row_above = None
            else:
                row_above = former_chunk[row-1]
            
            # determine what the row below should be
            if (row == num_rows-1):
                row_below = halo_below
            else:
                row_below = former_chunk[row+1]
        
        elif (my_rank == 0):  # bottom rank
            # determine what the row above should be
            if (row == 0):
                row_above = halo_above
            else:
                row_above = former_chunk[row-1]
            
            # determine what the row below should be
            if (row == num_rows-1):
                row_below = None
            else:
                row_below = former_chunk[row+1]
        
        else:  # middle rank
            # determine what the row above should be
            if (row == 0):
                row_above = halo_above
            else:
                row_above = former_chunk[row-1]
            
            # determine what the row below should be
            if (row == num_rows-1):
                row_below = halo_below
            else:
                row_below = former_chunk[row+1]
        
        neighbors = 0
        # left
        neighbors += former_chunk[row, 1]
        if (row_above is not None):
            neighbors += row_above[0] + row_above[1]
        
        if (row_below is not None):
            neighbors += row_below[0] + row_below[1]
        
        current_chunk[row, 0] = setCell(former_chunk[row, 0], neighbors)
        
        # middle
        for col in range(1, grid_size - 1):
            neighbors = former_chunk[row, col-1] + former_chunk[row, col+1]
            if (row_above is not None):
                neighbors += row_above[col-1] + row_above[col] + row_above[col+1]
            
            if (row_below is not None):
                neighbors += row_below[col-1] + row_below[col] + row_below[col+1]
            
            current_chunk[row, col] = setCell(former_chunk[row, col], neighbors)
        
        # right
        neighbors = former_chunk[row, grid_size-2]
        if (row_above is not None):
            neighbors += row_above[grid_size-2] + row_above[grid_size-1]
        
        if (row_below is not None):
            neighbors += row_below[grid_size-2] + row_below[grid_size-1]
        
        current_chunk[row, grid_size-1] = setCell(former_chunk[row, grid_size-1], neighbors)

    # Print the current_map if this is an output generation.
    if ((generation + 1) % output_generations == 0):
        # All the ranks send their current_chunk to rank 0, including rank 0 itself.
        # First I am initiating the comm.Ssend(). This is primarily to avoid a deadlock in rank 0.
        # Suppose that rank 0 calls comm.Ssend(), but it does not complete because it returns only after the
        # chunk has been recieved, and since it's sending it to itself, there is a deadlock.
        # Rank 0 has not yet called comm.Recv(), and in doing the comm.Ssend() there is no one on the other end
        # to recieve the message so it just hangs forever. Even worse, none of the other ranks can send
        # their chunk to rank 0 at this time.
        # comm.Issend() initiates the sending and then returns immediately, allowing rank 0 to then recieve that
        # message from itself, and recieve all the other chunks from the other processes.
        request = comm.Issend(buf=current_chunk, dest=0, tag=4)
        
        if (my_rank == 0):
            # Do the Gather in rank 0.
            for i in range(0, grid_size, num_rows):
                # default argument: Status status=None
                comm.Recv(buf=temp_chunk, source=i / num_rows, tag=4)
                
                col = i + num_rows-1
                while (col >= i):
                    grid[:, col] = temp_chunk[num_rows-1-col+i, :]
                    col -= 1
            
            print("Generation %d:" % (generation + 1), file=output, flush=True)
            for row in range(grid_size):
                for col in range(grid_size):
                    print("%d" % grid[row, col], end='', file=output, flush=True)
                print('', end='\n', file=output, flush=True)
        
        request.Wait()


# Stop the timer in each of the ranks.
local_finish = time.time()
# Calculate the elapsed time in each of the ranks.
local_elapsed = local_finish - local_start
# The global elapsed time (the time for the whole program) is the time it took the "slowest" process to finish,
# the maximum local_elapsed time.
elapsed = comm.reduce(sendobj=local_elapsed, op=MPI.MAX, root=0)

if (my_rank == 0):
    # The timing information is apended to the end of the file.
    print("%f" % round(elapsed, 6), file=timing, flush=True)

# Let the Python interpreter deallocate the halos.
halo_above = None
halo_below = None

# Let the Python interpreter deallocate the chunks.
chunk  = None
chunk2 = None

if (my_rank == 0 and grid_size is not None):
    temp_chunk = None
    output.close()
    timing.close()
    grid = None

# MPI.Finalize() is automatically called when the Python process exits.
