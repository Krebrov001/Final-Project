/*******************************************************************************
 * FILE: ethan_mm.cpp
 * AUTHOR: Ethan Robinson
 * PROJECT: Final Project, test matrix multiplication
 * CREATED: December 10th, 2019
 * 
 * DESCRIPTION: A C++ test implementation of matrix multiplication without MPI
 * 
 * LINKS:
 *  + How to preallocate a 2D matrix using arrays:
 *      https://www.geeksforgeeks.org/2d-vector-in-cpp-with-user-defined-size/
 *  +
 ******************************************************************************/

#include <iostream>
#include <fstream>
#include <ostream>
#include <vector>

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
    int matrix_size; //Matrix will be N * N

    //If 3 args, check if verbose flag
    if (argc == 3 && argv[1][0] == '-' && tolower(argv[1][1]) == 'v')
    {
        verbose = true;
        matrix_size = std::stoi(argv[2]);
    }
    else if (argc == 2)
    {
        matrix_size = std::stoi(argv[1]);
    }
    else //Invalid arguments
    {
        cerr << "Invalid arguments!" << endl;
        cerr
            << "Usage:"
            << endl
            << argv[0]
            << " <optionalFlag> <inputFilePath>"
            << endl;
    }

    //Check number of arguments
    if (argc < 2)
    {
        cerr << "Invalid number of arguments!" << endl;
        cerr
            << "Usage:"
            << endl
            << argv[0]
            << " <optionalFlag> <inputFilePath>"
            << endl;
        return EXIT_FAILURE;
    }

    //Matrix data for all processes
    const int N = matrix_size;

    //Master will store the matricies
    // std::vector<std::vector<float>> matrix_a;
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

    //Check matrix values
    if (verbose)
    {
        //Print matrix_a
        cout << "Matrix_A:" << endl;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                cout << "    " << matrix_a[i][j];
            }
            cout << endl;
        }

        //print matrix_b
        cout << "Matrix_B:" << endl;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                cout << "    " << matrix_b[i][j];
            }
            cout << endl;
        }
    }

    //Multiply matricies
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            matrix_result[i][j] = matrix_a[i][j] * matrix_b[i][j];
        }
    }

    //Cout results
    if (verbose)
    {
        cout << endl << "Result matrix:" << endl;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                cout << "    " << matrix_result[i][j];
            }
            cout << endl;
        }
    }

    return EXIT_SUCCESS;
}