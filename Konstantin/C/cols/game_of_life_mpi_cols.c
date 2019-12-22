#include <stdio.h>   /* for fopen(), fclose(), fread(), fscanf(), prinf(), fprintf(), fputc(), stderr */
#include <stdlib.h>  /* for exit(), EXIT_SUCCESS, EXIT_FAILURE */
#include <string.h>  /* for strlen(), strcpy(), strncpy() */
#include <mpi.h>     /* for MPI functions, etc */
#include <errno.h>   /* for perror() */
#include <stdint.h>  /* for uint8_t */
#include <ctype.h>   /* for isdigit() */


void* read_in_file(const char* const filename, int* grid_size, int* num_generations, int* output_generations, int num_processors)
{
    FILE* file = NULL;
    
    /* Enforce input file naming conventions. */
    
    if (strncmp(filename, "input", strlen("input")) != 0) {
        fprintf(stderr, "ERROR: the file name %s needs to be in the form inputN, where N is an integer.\n", filename);
        fprintf(stderr, "Example of a valid file name: input2\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    size_t index = strlen("input");
    while (filename[index] != '\0') {
        if (!isdigit(filename[index])) {
            fprintf(stderr, "ERROR: the file name %s needs to be in the form inputN, where N is an integer.\n", filename);
            fprintf(stderr, "Example of a valid file name: input2\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        ++index;
    }
    
    if ((file = fopen(filename, "r")) == NULL) {
        char* info_text = " could not be opened";
        size_t len1 = strlen(filename);
        size_t len2 = strlen(info_text);
        char message[len1 + len2 + 1];
        strncpy(message, filename, len1);
        strncpy(message + len1, info_text, len2);
        message[len1 + len2] = '\0';
        fprintf(stderr, "ERROR:\n");
        perror(message);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    fscanf(file, "%d %d %d\n", grid_size, num_generations, output_generations);
    
    if (*grid_size == -1 || *num_generations == -1 || *output_generations == -1) {
        char* info_text = " could not be read successfully";
        size_t len1 = strlen(filename);
        size_t len2 = strlen(info_text);
        char message[len1 + len2 + 1];
        strncpy(message, filename, len1);
        strncpy(message + len1, info_text, len2);
        message[len1 + len2] = '\0';
        fprintf(stderr, "ERROR: %s\n", message);
        fprintf(stderr, "File format:\n");
        fprintf(stderr, "N G O\n"
                        "NxN grid\n"
                        "\n"
                        "N - the size of the NxN grid of 0s and 1s\n"
                        "G - the number of generations to iterate through\n"
                        "O - the output generation value\n"
        );
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    if (*grid_size <= 0) {
        fprintf(stderr, "ERROR:\n");
        fprintf(stderr, "N - the size of the NxN grid must be > 0\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    if (*grid_size % 8 != 0) {
        fprintf(stderr, "ERROR:\n");
        fprintf(stderr, "N - the size of the NxN grid must be divisible by 8\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    if (*num_generations == 0) {
        fprintf(stderr, "ERROR:\n");
        fprintf(stderr, "G - the number of generations to iterate through must be > 0\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    if (*output_generations == 0) {
        fprintf(stderr, "ERROR:\n");
        fprintf(stderr, "O - the output generation value must be > 0\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    if (*num_generations % *output_generations != 0) {
        fprintf(stderr, "ERROR:\n");
        fprintf(stderr, "G - the number of generations to iterate through must be divisible by O - the output generation value\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    // You need to validate that the number of processors your code is being run with evenly divides the N or size of your grid.
    if (*grid_size % num_processors != 0) {
        fprintf(stderr, "ERROR:\n");
        fprintf(stderr, "N - the size of the NxN grid must be divisible by the number of processors %d\n", num_processors);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    uint8_t (* const map)[*grid_size] = (uint8_t (* const)[*grid_size]) malloc( sizeof(uint8_t[*grid_size][*grid_size]) );
    if (map == NULL) {
        // malloc() sets errno to ENOMEM upon failure.
        perror("ERROR: Unable to read the grid into memory");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    for (int row = 0; row < *grid_size; ++row) {
        uint8_t dummy;
        fread(map[row], sizeof(uint8_t), *grid_size, file);
        fread(&dummy, sizeof(uint8_t), 1, file);  // consume the '\n'
    }
    
    for (int row = 0; row < *grid_size; ++row) {
        for (int col = 0; col < *grid_size; ++col) {
            map[row][col] -= '0';  // convert from char to int
        }
    }
    
    fclose(file);
    file = NULL;
    
    return map;
}


static inline uint8_t setCell(uint8_t former_cell, int neighbors)
{
    if (former_cell == 1) {
        if (neighbors < 2 || neighbors > 3) {
            return 0;
        } else {
            return 1;
        }
    } else {  // former_cell == 0
        if (neighbors == 3) {
            return 1;
        } else {
            return 0;
        }
    }
}


// This macro defines a type cast that allows a pointer to be used as a matrix, with width of size.
// Meaning you can use Matrix(pointer, size)[4][5] to access the element in the 4th row and 5th column
// of that matrix.
// pointer should be any valid pointer
// size should be an integer
// Using any other data types for these two arguments is undefined.
#define Matrix(pointer, size) ((uint8_t (* const)[size])pointer)
// Similarly for a single row.
#define Row(pointer) ((uint8_t * const) pointer)

const int MAX_STRING = 100;

int main(int argc, char** argv) {
    int  comm_sz;   /* Number of processes      */
    int  my_rank;   /* My process rank          */
    // In the rank 0, grid points to a dynamic matrix containing the entire map.
    // In the other ranks, it is a NULL pointer.
    void* grid = NULL;
    // In all the ranks, including 0, this is the chunk (row or column) that belongs
    // to the current process.
    void* chunk = NULL;
    // In all the ranks, including 0, this is the chunk (row or column) that belongs
    // to the current process.
    // chunk and chunk2 alternate between themselves as the current_chunk and the former_chunk.
    void* chunk2 = NULL;
    // In rank 0, this is NULL.
    // In all other ranks, this is a copy of the lowest row in the rank above it.
    void* halo_above = NULL;
    // In rank comm_sz-1 (the lowest rank), this is NULL.
    // In all other ranks, this is a copy of the highest row in the rank below it.
    void* halo_below = NULL;
    void* row_above = NULL;
    void* row_below = NULL;
    // In the rank 0, this points to a temporary chunk which represents a column-wise sectioning of the grid,
    // instead of a row-wise sectioning of the grid. Although this temp_chunk is looking like a row,
    // it is in fact a rotated column of the original grid, which is given to a rank for processing.
    void* temp_chunk = NULL;
    // Because the mpirun executable itself generates standard error and standard output,
    // I need to create a new file / output stream where the application's output should be written.
    FILE* output = NULL;
    // Similarly, this is the file / output stream where the application's timing information shoudl be written.
    FILE* timing = NULL;

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (argc != 2) {
        // If incorrect arguments supplied, have rank 0 abort all the processes.
        // You always want to make sure that only one rank aborts all the processes.
        // The other processes do MPI_Recv() for a message that never comes, so they are blocked.
        // If I would have put MPI_Abort() inside the other processes as well, one of them could
        // potentially kill rank 0 before it had a chance to fprintf() the message out to the screen.
        // A single MPI_Abort() terminates all the processes.
        if (my_rank == 0) {
            fprintf(stderr, "Usage:\n$ %s <input_file>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        } else {
            int dummy;
            MPI_Recv(&dummy, 1, MPI_INT, MPI_ANY_SOURCE, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    const char* const input_filename = argv[1];

    int grid_size = -1;
    int num_generations = -1;
    int output_generations = -1;

    if (my_rank == 0) {
        // read in the file
        grid = read_in_file(input_filename, &grid_size, &num_generations, &output_generations, comm_sz);

        // send the data to all processes.
        for (int i = 0; i < comm_sz; ++i) {
            // MPI_Bsend() is used to send multiple data in parallel.
            // If I would use MPI_Ssend(), then rank 0 would wait on every single other rank, and give send them data
            // one by one with confirmation. However if I use MPI_Bsend(), then it does not wait for a confirmation,
            // but it sends the data at once, and then the other rank recieves it when it is ready, and the rank 0
            // does not have to wait on it. MPI_Send() could be either MPI_Bsend() or MPI_Ssend(), and this is
            // implementation dependent.
            MPI_Bsend(&grid_size, 1, MPI_INT, i, 0, MPI_COMM_WORLD);  // tag 0 is for grid_size
            MPI_Bsend(&num_generations, 1, MPI_INT, i, 1, MPI_COMM_WORLD);  // tag 1 is for num_generations
            MPI_Bsend(&output_generations, 1, MPI_INT, i, 2, MPI_COMM_WORLD);  // tag 2 is for output_generations
        }
    } else {
        // Recieving the data is used as a synchronization mechanism for the other processes.
        // Rank 0 reads in the file, which takes a lot of time. I do not want my other processes to run off and leave
        // rank 0 behind. By doing the MPI_Recv(), they are patiently waiting with the open mouths until rank 0 finally
        // finishes reading in the file and sends them their data.
        //
        // I am using a tagging system to distinguish the data. Rank 0 uses MPI_Bsend() to send the data.
        // There is a slight chance that the messages could be recieved in a different order than they are sent.
        // If that is the case, explicitly provide tags to recieve the messages in the same order that they were sent.
        // There is no efficiency loss to this, but it prevents some bugs that might be caused when using the same tag
        // for all your sent and recieved data.
        MPI_Recv(&grid_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // tag 0 is for grid_size
        MPI_Recv(&num_generations, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // tag 1 is for num_generations
        MPI_Recv(&output_generations, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);  // tag 2 is for output_generations
    }
    
    // Open up the output file/stream.
    if (my_rank == 0) {
        // This code creates the output filename based on the input filename.
        // input_filename    output_filename
        // input1            output1
        // input11           output11
        //
        // +1 for the 'o' in "output", +1 for the '\0'
        size_t input_string_len = strlen(input_filename);
        char output_filename[input_string_len + 2];
        output_filename[input_string_len + 1] = '\0';
        // offset is set to index to the start of the digit sequence at the end.
        int offset = input_string_len - 1;
        while (isdigit(input_filename[offset-1])) {
            --offset;
        }
        strcpy(output_filename + offset + 1, input_filename + offset);
        strncpy(output_filename, "output", strlen("output"));
        // Try to open the output stream.
        if ((output = fopen(output_filename, "w")) == NULL) {
            fprintf(stderr, "ERROR: Could not open the file %s for writing output.\n", output_filename);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        // Success!
        } else {
            char dummy = 'A';
            // Send the success message to all processes.
            for (int i = 0; i < comm_sz; ++i) {
                MPI_Send(&dummy, 1, MPI_CHAR, i, 100, MPI_COMM_WORLD);
            }
        }
    // Make the other ranks either wait for a success message to be sent, or be killed upon failure.
    // This is used as a synchronization device.
    } else {
        char dummy;
        MPI_Recv(&dummy, 1, MPI_CHAR, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // Open up the timing file/stream.
    if (my_rank == 0) {
        // This code creates the timing filename based on the input filename.
        // input_filename    timing_filename
        // input1            timing1
        // input11           timing11
        //
        // +1 for the 't' in "timing", +1 for the '\0'
        size_t input_string_len = strlen(input_filename);
        char timing_filename[input_string_len + 2];
        timing_filename[input_string_len + 1] = '\0';
        // offset is set to index to the start of the digit sequence at the end.
        int offset = input_string_len - 1;
        while (isdigit(input_filename[offset-1])) {
            --offset;
        }
        strcpy(timing_filename + offset + 1, input_filename + offset);
        strncpy(timing_filename, "timing", strlen("timing"));
        // Try to open the timing stream.
        // It should be opened in the append mode, because the MPI executable may be run numerous times,
        // and the timing information for each run be appended to the timing file, which contains the timing information
        // of all the runs of the program.
        if ((timing = fopen(timing_filename, "a")) == NULL) {
            fprintf(stderr, "ERROR: Could not open the file %s for writing timing information.\n", timing_filename);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        // Success!
        } else {
            char dummy = 'A';
            // Send the success message to all processes.
            for (int i = 0; i < comm_sz; ++i) {
                MPI_Send(&dummy, 1, MPI_CHAR, i, 101, MPI_COMM_WORLD);
            }
        }
    // Make the other ranks either wait for a success message to be sent, or be killed upon failure.
    // This is used as a synchronization device.
    } else {
        char dummy;
        MPI_Recv(&dummy, 1, MPI_CHAR, 0, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // These variables are relevant for timing information.
    // Each rank gets it's own copy of these variables,
    // but only the local ones are relevant to a single rank.
    // And double elapsed is only relevant to rank 0.
    // It contains the total elapsed time for the whole entire MPI program,
    // including the running time of all the ranks.
    double local_start = 0.0,
           local_finish = 0.0,
           local_elapsed = 0.0,
           elapsed = 0.0;
    
    // Makes all the ranks pause and wait for each other before continuing down.
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Start the timer in each of the ranks.
    local_start = MPI_Wtime();
    
    // Setup the chunks.
    int chunk_size = grid_size * grid_size / comm_sz;  // the number of elements in a chunk
    int num_rows = chunk_size / grid_size;  // the number of rows in a chunk
    chunk = malloc( sizeof(uint8_t[num_rows][grid_size]) );
    chunk2 = malloc( sizeof(uint8_t[num_rows][grid_size]) );
    
    // Setup the halos.
    if (my_rank != comm_sz - 1) {
        halo_above = malloc( sizeof(uint8_t[grid_size]) );
    }
    if (my_rank != 0) {
        halo_below = malloc( sizeof(uint8_t[grid_size]) );
    }
    
    // Setup the temp chunk.
    if (my_rank == 0) {
        // This temp_chunk is in fact a rotated version of the original chunk.
        // So it's size should be viewed as uint8_t[grid_size][num_rows].
        temp_chunk = malloc( sizeof(uint8_t[num_rows][grid_size]) );
    }
    
    /* This code simulates a column-wise Scatter. */
    if (my_rank == 0) {
        // rank 0 sends a chunk to all other ranks, including itself.
        // In fact, it first sends the chunk to itself.
        // However, I am using MPI_Ssend(), which only returns after the message has been recieved
        // by the other end. If there is no recieve at the other end, there is a deadlock.
        // So I need to call MPI_Recv() before calling the MPI_Send(), but then MPI_Recv() only returns
        // after the message has been recieved by itself, and because no one has sent anything yet,
        // it would also be a deadlock. I need to initiate a MPI_Irecv() to get rid of the deadlock.
        MPI_Request request;
        MPI_Irecv(chunk, sizeof(uint8_t[num_rows][grid_size]), MPI_UINT8_T, 0, 4, MPI_COMM_WORLD, &request);
        
        // This outer loop iterates through all the column-wise chunks in the original grid.
        // For each rank, fill in a temp_chunk, and send it over.
        for (int i = 0; i < grid_size; i += num_rows) {
            // This inner loop iterates though a single column-wise chunk in the original grid
            // and assigns that corresponding element into the row-wise temp_chunk.
            // The locality of iteration is made in favor of the temp_chunk.
            for (int col = i + num_rows-1; col >= i; --col) {
                for (int row = 0; row < grid_size; ++row) {
                    Matrix(temp_chunk, grid_size)[num_rows-1-col+i][row] = Matrix(grid, grid_size)[row][col];
                }
            }
            // Send the temp_chunk to the chunk in the corresponding rank.
            // I am using MPI_Ssend() to be sure that a given chunk has been recieved by the destination rank,
            // before overwriting the temp_chunk and sending out the next one.
            // MPI_Bsend() will not work because for large chunk sizes, a copying buffer has to be manually
            // allocated by the programmer, and we have no way of knowing when that message has been sent in
            // order to call MPI_Bsend() again and overwrite the copying buffer.
            // tag 4 is for messages feigning collective communications.
            MPI_Ssend(temp_chunk, sizeof(uint8_t[num_rows][grid_size]), MPI_UINT8_T, i / num_rows, 4, MPI_COMM_WORLD);
        }
        // Complete the MPI_Recv() initiated up above.
        MPI_Wait(&request, MPI_STATUS_IGNORE);
        
    } else {
        // Recieve your chunk from rank 0.
        MPI_Recv(chunk, sizeof(uint8_t[num_rows][grid_size]), MPI_UINT8_T, 0, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    
    // Copy the scattered data from the chunk into chunk2.
    for (int row = 0; row < num_rows; ++row) {
        for (int col = 0; col < grid_size; ++col) {
            Matrix(chunk2, grid_size)[row][col] = Matrix(chunk, grid_size)[row][col];
        }
    }
    
    uint8_t (* current_chunk)[grid_size] = NULL;
    uint8_t (* former_chunk)[grid_size] = NULL;
    
    for (int generation = 0; generation < num_generations; ++generation) {
        if (generation % 2 == 0) {
            current_chunk = chunk2;
            former_chunk = chunk;
        } else {
            current_chunk = chunk;
            former_chunk = chunk2;
        }
        
        /* The halo arrays get updated each iteration. */
        
        MPI_Request request1;
        MPI_Request request2;
        // If you have the below buffer.
        if (halo_below != NULL) {
            // Send the bottom row in the chunk to the rank directly below you.
            MPI_Issend(&former_chunk[num_rows-1][0], grid_size, MPI_UINT8_T, my_rank-1, 3, MPI_COMM_WORLD, &request1);
        }
        // If you have the above buffer.
        if (halo_above != NULL) {
            // Send the top row in the chunk to the rank directly above you.
            MPI_Issend(&former_chunk[0][0], grid_size, MPI_UINT8_T, my_rank+1, 3, MPI_COMM_WORLD, &request2);
        }
        // If you have the below buffer.
        if (halo_below != NULL) {
            // Recieve the top row in the chunk from the rank directly below you.
            MPI_Recv(halo_below, grid_size, MPI_UINT8_T, my_rank-1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // If you have the above buffer.
        if (halo_above != NULL) {
            // Recieve the bottom row in the chunk from the rank directly above you.
            MPI_Recv(halo_above, grid_size, MPI_UINT8_T, my_rank+1, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // If you have the below buffer.
        if (halo_below != NULL) {
            MPI_Wait(&request1, MPI_STATUS_IGNORE);
        }
        // If you have the above buffer.
        if (halo_above != NULL) {
            MPI_Wait(&request2, MPI_STATUS_IGNORE);
        }
    
        // Process the data.
        // This loops through all the rows.
        // In a single iteration of this loop, an entire row of cells, with all the columns, is computed.
        // Before performing the actual Game of Life Algorithm, the rows above and below that cell are pre-computed
        // in order to make it easier.
        for (int row = 0; row < num_rows; ++row) {
            if (my_rank == comm_sz-1) {  // top rank
                // determine what the row above should be
                if (row == 0) {
                    row_above = NULL;
                } else {
                    row_above = &former_chunk[row-1][0];
                }
                // determine what the row below should be
                if (row == num_rows-1) {
                    row_below = halo_below;
                } else {
                    row_below = &former_chunk[row+1][0];
                }
            } else if (my_rank == 0) {  // bottom rank
                // determine what the row above should be
                if (row == 0) {
                    row_above = halo_above;
                } else {
                    row_above = &former_chunk[row-1][0];
                }
                // determine what the row below should be
                if (row == num_rows-1) {
                    row_below = NULL;
                } else {
                    row_below = &former_chunk[row+1][0];
                }
            } else {  // middle rank
                // determine what the row above should be
                if (row == 0) {
                    row_above = halo_above;
                } else {
                    row_above = &former_chunk[row-1][0];
                }
                // determine what the row below should be
                if (row == num_rows-1) {
                    row_below = halo_below;
                } else {
                    row_below = &former_chunk[row+1][0];
                }
            }
            
            int neighbors = 0;
            // left
            neighbors += former_chunk[row][1];
            if (row_above != NULL) {
                neighbors += Row(row_above)[0] + Row(row_above)[1];
            }
            if (row_below != NULL) {
                neighbors += Row(row_below)[0] + Row(row_below)[1];
            }
            current_chunk[row][0] = setCell(former_chunk[row][0], neighbors);
            
            // middle
            for (int col = 1; col < grid_size - 1; ++col) {
                neighbors = former_chunk[row][col-1] + former_chunk[row][col+1];
                if (row_above != NULL) {
                    neighbors += Row(row_above)[col-1] + Row(row_above)[col] + Row(row_above)[col+1];
                }
                if (row_below != NULL) {
                    neighbors += Row(row_below)[col-1] + Row(row_below)[col] + Row(row_below)[col+1];
                }
                current_chunk[row][col] = setCell(former_chunk[row][col], neighbors);
            }
            
            // right
            neighbors = former_chunk[row][grid_size-2];
            if (row_above != NULL) {
                neighbors += Row(row_above)[grid_size-2] + Row(row_above)[grid_size-1];
            }
            if (row_below != NULL) {
                neighbors += Row(row_below)[grid_size-2] + Row(row_below)[grid_size-1];
            }
            current_chunk[row][grid_size-1] = setCell(former_chunk[row][grid_size-1], neighbors);
        
        }
        
        // Print the current_map if this is an output generation.
        if ((generation + 1) % output_generations == 0) {
            // All the ranks send their current_chunk to rank 0, including rank 0 itself.
            // First I am initiating the MPI_Ssend(). This is primarily to avoid a deadlock in rank 0.
            // Suppose that rank 0 calls MPI_Ssend(), but it does not complete because it returns only after the
            // chunk has been recieved, and since it's sending it to itself, there is a deadlock.
            // Rank 0 has not yet called MPI_Recv(), and in doing the MPI_Ssend() there is no one on the other end
            // to recieve the message so it just hangs forever. Even worse, none of the other ranks can send
            // their chunk to rank 0 at this time.
            // MPI_Issend() initiates the sending and then returns immediately, allowing rank 0 to then recieve that
            // message from itself, and recieve all the other chunks from the other processes.
            MPI_Request request;
            MPI_Issend(current_chunk, sizeof(uint8_t[num_rows][grid_size]), MPI_UINT8_T, 0, 4, MPI_COMM_WORLD, &request);
        
            if (my_rank == 0) {
                // Do the Gather in rank 0.
                for (int i = 0; i < grid_size; i += num_rows) {
                    MPI_Recv(temp_chunk, sizeof(uint8_t[num_rows][grid_size]), MPI_UINT8_T, i / num_rows, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    
                    for (int col = i + num_rows-1; col >= i; --col) {
                        for (int row = 0; row < grid_size; ++row) {
                            Matrix(grid, grid_size)[row][col] = Matrix(temp_chunk, grid_size)[num_rows-1-col+i][row];
                        }
                    }
                }
                
                fprintf(output, "Generation %d:\n", generation + 1);
                for (int row = 0; row < grid_size; ++row) {
                    for (int col = 0; col < grid_size; ++col) {
                        fprintf(output, "%hhu", Matrix(grid, grid_size)[row][col]);
                    }
                    fputc('\n', output);
                }
            }
            
            MPI_Wait(&request, MPI_STATUS_IGNORE);
        }
    
    }
    
    // Stop the timer in each of the ranks.
    local_finish = MPI_Wtime();
    // Calculate the elapsed time in each of the ranks.
    local_elapsed = local_finish - local_start;
    // The global elapsed time (the time for the whole program) is the time it took the "slowest" process to finish,
    // the maximum local_elapsed time.
    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (my_rank == 0) {
        // The timing information is apended to the end of the file.
        fprintf(timing, "%.6f\n", elapsed);
    }
    
    // Deallocate the halos.
    if (halo_above != NULL) {
        free(halo_above);
    }
    if (halo_below != NULL) {
        free(halo_below);
    }
    
    // Deallocate the chunks.
    free(chunk);
    chunk = NULL;
    free(chunk2);
    chunk2 = NULL;
    
    if (my_rank == 0 && grid != NULL) {
        free(temp_chunk);
        fclose(output);
        fclose(timing);
        free(grid);
    }
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}
