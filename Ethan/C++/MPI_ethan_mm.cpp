/*******************************************************************************
 * FILE: ethan_mm.cpp
 * AUTHOR: Ethan Robinson
 * PROJECT: Final Project, openMPI matrix multiplication
 * CREATED: December 10th, 2019
 * 
 * DESCRIPTION: A C++ implementation of matrix multiplication for a jetson cluster
 ******************************************************************************/

#include <iostream>
#include <fstream>
#include <ostream>
#include <vector>
#include <mpi.h>
#include <math.h>

#define MASTER 0

//Used for MPI send/recv tags
#define MATRIX_A 0
#define MATRIX_B 5000
#define RESULT 10000

using std::cerr;
using std::cin;
using std::cout;
using std::endl;
using std::string;

//Set to true for verbose output
bool verbose = false;

int main(int argc, char *argv[])
{
    //Local Vars
    int world_size; //Number of processes
    int world_rank; //Process identifier.
    int N;          //Matrix will be N * N

    //Initalize MPI stuff
    MPI_Init(NULL, NULL);
    //Get number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    //Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    //Make sure there is more than 1 process
    if (world_size < 2)
    {
        cerr << "World size must be greater than 1" << endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    //If 3 args, check if verbose flag
    if (argc == 3 && argv[1][0] == '-' && tolower(argv[1][1]) == 'v')
    {
        verbose = true;
        N = std::stoi(argv[2]);
    }
    else if (argc == 2)
    {
        N = std::stoi(argv[1]);
    }
    else //Invalid arguments
    {
        /* Rank 0 should abort to kill all other processes, and the other ranks
         * should call MPI_Recv to sleep while waiting to be aborted.
         */
        if (world_rank == 0)
        {
            cerr << "Invalid arguments!" << endl;
            cerr
                << "Usage:"
                << endl
                << argv[0]
                << " <optionalFlag> <inputFilePath>"
                << endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        else
        {
            int temp;
            //Block to sleep while waiting for an MPI_Abort
            MPI_Recv(&temp, 1, MPI_INT, MPI_ANY_SOURCE, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    //Check number of arguments
    if (argc < 2)
    {
        /* Rank 0 should abort to kill all other processes, and the other ranks
         * should call MPI_Recv to sleep while waiting to be aborted.
         */
        if (world_rank == 0)
        {
            cerr << "Invalid number of arguments!" << endl;
            cerr
                << "Usage:"
                << endl
                << argv[0]
                << " <optionalFlag> <inputFilePath>"
                << endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        else
        {
            int temp;
            //Block to sleep while waiting for an MPI_Abort
            MPI_Recv(&temp, 1, MPI_INT, MPI_ANY_SOURCE, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    //Minimum matrix size is world size
    if (N < world_size)
    {
        if (world_rank == 0)
        {
            cerr << "Minimum maxtrix size is 8" << endl;
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        else
        {
            int temp;
            //Block to sleep while waiting for an MPI_Abort
            MPI_Recv(&temp, 1, MPI_INT, MPI_ANY_SOURCE, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    //Matrix data for all processes
    double temp = N / (world_size - 1);
    const int chunk_size = floor(temp);                        // # of rows that will evenly divide into workers
    const int remainder = N - (chunk_size * (world_size - 1)); //Remainder rows

    //------------------------------Master (rank_0)-----------------------------
    if (world_rank == 0)
    {
        double start = MPI_Wtime();
        //Master will store the matricies
        std::vector<std::vector<float>> matrix_a(N, std::vector<float>(N));
        std::vector<std::vector<float>> matrix_b(N, std::vector<float>(N));
        std::vector<std::vector<float>> matrix_result(N, std::vector<float>(N));

        //Randomly gnerate matricies. Numbers range from -50 to 50
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                matrix_a[i][j] = (float)rand() / (float)(RAND_MAX / 100) - 50;
                matrix_b[i][j] = (float)rand() / (float)(RAND_MAX / 100) - 50;
                matrix_result[i][j] = 0.0f;
            }
        }

        cout << "Matrix_A:" << endl;
        for (int i = 0; i < N; i++)
        {
            cout << i << '\t';
            for (int j = 0; j < N; j++)
            {
                cout << matrix_a[i][j] << "    ";
            }
            cout << endl;
        }

        cout << "Matrix_B:" << endl;
        for (int i = 0; i < N; i++)
        {
            cout << i << '\t';
            for (int j = 0; j < N; j++)
            {
                cout << matrix_b[i][j] << "    ";
            }
            cout << endl;
        }
        cout << endl;

        int rowIndex = 0;
        std::vector<float> sendMe;

        //Send chunks of Matrix A to workers
        for (int dest = 1; dest < world_size; dest++)
        {
            for (int i = 0; i < chunk_size; i++)
            {
                rowIndex = (i) + ((dest - 1) * chunk_size);
                sendMe = matrix_a[rowIndex];

                MPI_Send(
                    &sendMe[0],          //temp vector
                    N,                   //Number of elements
                    MPI_FLOAT,           //Type
                    dest,                //Destination
                    MATRIX_A + rowIndex, //Tag
                    MPI_COMM_WORLD);
            }
            if (remainder && dest == (world_size - 1))
            {
                //Send remainder
                for (int i = 0; i < remainder; i++)
                {
                    rowIndex = (i) + (dest * chunk_size);
                    sendMe = matrix_a[rowIndex];

                    MPI_Send(
                        &sendMe[0],          //temp vector
                        N,                   //Number of elements
                        MPI_FLOAT,           //Type
                        dest,                //Destination
                        MATRIX_A + rowIndex, //Tag
                        MPI_COMM_WORLD);
                }
            }
        }

        //Send chunks of Matrix B to workers
        for (int dest = 1; dest < world_size; dest++)
        {
            for (int i = 0; i < chunk_size; i++)
            {
                rowIndex = (i) + ((dest - 1) * chunk_size);
                sendMe = matrix_b[rowIndex];

                MPI_Send(
                    &sendMe[0],          //temp vector
                    N,                   //Number of elements
                    MPI_FLOAT,           //Type
                    dest,                //Destination
                    MATRIX_B + rowIndex, //Tag
                    MPI_COMM_WORLD);
            }
            if (remainder && dest == (world_size - 1))
            {
                //Send remainder
                for (int i = 0; i < remainder; i++)
                {
                    rowIndex = (i) + (dest * chunk_size);
                    sendMe = matrix_b[rowIndex];

                    MPI_Send(
                        &sendMe[0],          //temp vector
                        N,                   //Number of elements
                        MPI_FLOAT,           //Type
                        dest,                //Destination
                        MATRIX_B + rowIndex, //Tag
                        MPI_COMM_WORLD);
                }
            }
        }

        //Recieve processed chunks and store them into matrix results
        for (int worker = 1; worker < world_size; worker++)
        {
            //Recieve a chunk
            for (int i = 0; i < chunk_size; i++)
            {
                rowIndex = (i) + ((worker - 1) * chunk_size);
                MPI_Recv(
                    &sendMe[0],
                    N,
                    MPI_FLOAT,
                    worker,
                    (RESULT + rowIndex),
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
                matrix_result[rowIndex] = sendMe;
            }

            //If remainder, recieve those too
            if (remainder && worker == (world_size - 1))
            {
                for (int i = 0; i < remainder; i++)
                {
                    rowIndex = (i) + (worker * chunk_size);
                    MPI_Recv(
                        &sendMe[0],
                        N,
                        MPI_FLOAT,
                        worker,
                        (RESULT + rowIndex),
                        MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE);

                    matrix_result[rowIndex] = sendMe;
                }
            }
        }

        cout << "FINAL RESULT:" << endl;
        for (int i = 0; i < N; i++)
        {
            cout << i << '\t';
            for (int j = 0; j < N; j++)
            {
                cout << matrix_result[i][j] << "    ";
            }
            cout << endl;
        }
        cout << endl;

        //Store timing information in a text file with size N in title
        double finish = MPI_Wtime();
        double elapsed = finish - start;
        string timeFile = "time_";
        timeFile = timeFile + std::to_string(N) + ".txt";
        std::ofstream timingOut;
        timingOut.open(timeFile.c_str(), std::ios_base::app );
        timingOut << elapsed << endl;
        timingOut.close();
    }
    //----------------------------Workers (rank != 0)-----------------------------
    else
    {
        //Local vectors for workers
        std::vector<std::vector<float>> worker_matrix_a(chunk_size, std::vector<float>(N));
        std::vector<std::vector<float>> worker_matrix_b(chunk_size, std::vector<float>(N));
        std::vector<std::vector<float>> worker_result(chunk_size, std::vector<float>(N));

        int rowIndex;

        //recieve matrix_a chunk
        for (int i = 0; i < chunk_size; i++)
        {
            rowIndex = (i) + ((world_rank - 1) * chunk_size);
            MPI_Recv(
                &worker_matrix_a[i][0],
                N,
                MPI_FLOAT,
                MASTER,
                (MATRIX_A + rowIndex),
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
        }
        if (verbose)
        {
            for (int i = 0; i < chunk_size; i++)
            {
                rowIndex = (i) + ((world_rank - 1) * chunk_size);

                cout << "A_W" << world_rank << '[' << rowIndex << ']' << '\t';
                for (int j = 0; j < N; j++)
                {
                    cout << worker_matrix_a[i][j] << "    ";
                }
                cout << endl;
            }
        }

        //recieve matrix_b chunk
        for (int i = 0; i < chunk_size; i++)
        {
            rowIndex = (i) + ((world_rank - 1) * chunk_size);
            MPI_Recv(
                &worker_matrix_b[i][0],
                N,
                MPI_FLOAT,
                MASTER,
                (MATRIX_B + rowIndex),
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
        }
        if (verbose)
        {
            for (int i = 0; i < chunk_size; i++)
            {
                cout << "B_W" << world_rank << '[' << rowIndex << ']' << '\t';
                for (int j = 0; j < N; j++)
                {
                    cout << worker_matrix_b[i][j] << "    ";
                }
                cout << endl;
            }
        }

        //Multiply matricies and store in worker_result
        for (int i = 0; i < chunk_size; i++)
        {
            for (int j = 0; j < N; j++)
            {
                worker_result[i][j] = worker_matrix_a[i][j] * worker_matrix_b[i][j];
            }
        }

        //Send worker_result to master
        std::vector<float> sendMe;
        for (int i = 0; i < chunk_size; i++)
        {
            rowIndex = (i) + ((world_rank - 1) * chunk_size);
            sendMe = worker_result[i];
            MPI_Send(
                &sendMe[0],
                N,
                MPI_FLOAT,
                MASTER,
                (RESULT + rowIndex),
                MPI_COMM_WORLD);
        }

        //-------------------------------Last worker recieves the remainder rows
        if (remainder && world_rank == (world_size - 1))
        {
            std::vector<std::vector<float>> remainder_worker_matrix_a(remainder, std::vector<float>(N));
            std::vector<std::vector<float>> remainder_worker_matrix_b(remainder, std::vector<float>(N));
            std::vector<std::vector<float>> remainder_worker_result(remainder, std::vector<float>(N));

            //Recieve matrix_a remainder
            for (int i = 0; i < remainder; i++)
            {
                rowIndex = (i) + (world_rank * chunk_size);
                MPI_Recv(
                    &remainder_worker_matrix_a[i][0],
                    N,
                    MPI_FLOAT,
                    MASTER,
                    (MATRIX_A + rowIndex),
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);

                if (verbose)
                {
                    cout << "<RMDR>A_W" << world_rank << '[' << rowIndex << ']' << '\t';
                    for (int j = 0; j < N; j++)
                    {
                        cout << remainder_worker_matrix_a[i][j] << "    ";
                    }
                    cout << endl;
                }
            }

            //Recieve matrix_b remainder
            for (int i = 0; i < remainder; i++)
            {
                rowIndex = (i) + (world_rank * chunk_size);
                MPI_Recv(
                    &remainder_worker_matrix_b[i][0],
                    N,
                    MPI_FLOAT,
                    MASTER,
                    (MATRIX_B + rowIndex),
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);

                if (verbose)
                {
                    cout << "<RMDR>B_W" << world_rank << '[' << rowIndex << ']' << '\t';
                    for (int j = 0; j < N; j++)
                    {
                        cout << remainder_worker_matrix_b[i][j] << "    ";
                    }
                    cout << endl;
                }
            }

            //Multiply remainders
            for (int i = 0; i < remainder; i++)
            {
                for (int j = 0; j < N; j++)
                {
                    remainder_worker_result[i][j] = remainder_worker_matrix_a[i][j] * remainder_worker_matrix_b[i][j];
                }
            }

            //Send remainders
            for (int i = 0; i < remainder; i++)
            {
                rowIndex = (i) + (world_rank * chunk_size);
                sendMe = remainder_worker_result[i];
                MPI_Send(
                    &sendMe[0],
                    N,
                    MPI_FLOAT,
                    MASTER,
                    (RESULT + rowIndex),
                    MPI_COMM_WORLD);
            }
        }
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}