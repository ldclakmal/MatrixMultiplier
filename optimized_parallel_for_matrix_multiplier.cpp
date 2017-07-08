/* File:
 *     optimized_parallel_for_matrix_multiplier.cpp
 *
 *
 * Purpose:
 *     Computes a optimzed version of matrix-matrix multiplication
 *     using parallel for loops.
 *     Matrix size changes from 200 to 2000 in steps of 200.
 *
 * Input:
 *     None unless compiled with DEBUG flag.
 *
 * Output:
 *     Elapsed time for the computation for each size of matrix.
 *
 * Compile:
 *    g++ -fopenmp -lgomp -std=c++11 optimized_parallel_for_matrix_multiplier.cpp -o out
 *
 * Usage:
 *    ./out
 *
 */

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <random>

using namespace std;

// retuen the transpose of a given nxn matrix
inline void transpose(double **matA, double **matB, int n, int block_size) {

#pragma omp parallel
    {
        int_fast16_t i, j;								// Use fast_int to make the process quick
#pragma omp for                                          // Make the first loop parallel
        for (i = 0; i < n; i += block_size) {
            for (j = 0; j < n; j+= block_size) {
                for (size_t k = 0; k < block_size; k++)			// Use cache blocking to make the transpose quickly
                {
                    for (size_t l = 0; l < block_size; l++)
                    {
                        matB[j+l][i+k] = matA[i+k][j+l];
                    }
                }

            }
        }
    }
}

//Return the dot product of two vectors
inline double get_vector_dot_product(double *vecA, double *vecB, int n) {

    double temp = 0.0;

    for (int_fast16_t i = 0; i < n / 8; i++) {
        temp += vecA[8 * i] * vecB[8 * i]
                + vecA[8 * i + 1] * vecB[8 * i + 1]
                + vecA[8 * i + 2] * vecB[8 * i + 2]
                + vecA[8 * i + 3] * vecB[8 * i + 3]						// Loop unrolling of 8 for fast execution
                + vecA[8 * i + 4] * vecB[8 * i + 4]
                + vecA[8 * i + 5] * vecB[8 * i + 5]
                + vecA[8 * i + 6] * vecB[8 * i + 6]
                + vecA[8 * i + 7] * vecB[8 * i + 7];
    }


    return temp;

}

// Return the multiplication of given matrices using optimized techniques
void get_parallel_matrix_mulT(double **matA, double **matB, double **matBT, double **matC, int n, int block_size)
{

    transpose(matB, matBT, n, block_size);								// Take the transpose of the matrix
#pragma omp parallel
    {
        int_fast16_t i, j, k;
#pragma omp for                                       // Make the first loop parallel
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                matC[i][j] = get_vector_dot_product(matA[i], matBT[j], n);		// Blocking and taking dot product
            }
        }

    }
}

// Return a random double number between given two double values
double fRand(double fMin, double fMax)
{
    random_device rd;										// Seed
    mt19937 mt(rd());										// Random number generator
    uniform_real_distribution<double> dist(fMin, fMax);		// Distribution of numbers

    return dist(mt);
}

// Populate given 2-d arrays using random double values
void populate_array(double **matA, double **matB, int n)
{
    int i = 0, j;

    for (i = 0; i<n; i++) {
        for (j = 0; j < n; j++)
        {
            matA[i][j] = fRand(1.0, 2000.0);
            matB[i][j] = fRand(1.0, 2000.0);
        }

    }

}

void visualize_matrix(double **mat, int n)
{
    int i = 0, j = 0;

    for (i = 0; i<n; i++) {
        for (j = 0; j < n; j++)
        {
            cout << mat[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

//Returns the mean of set of floats
double get_mean(double *execution_times, int num_execution_times) {
    double sum = 0.0f;
    for (int i = 0; i < num_execution_times; i++) {
        sum += execution_times[i];
    }
    return sum / (double)num_execution_times;
}

//Returns the standard deviation of set of floats
double get_standard_diviation(double *execution_times, double mean, int num_execution_times) {
    double sum = 0.0f;
    for (int i = 0; i < num_execution_times; i++) {
        sum += (execution_times[i] - mean)*(execution_times[i] - mean);
    }
    return (double)sqrt(sum / (double)(num_execution_times - 1));
}

//Returns required number of executions, provided mean and standard deviation
int get_num_execution(double mean, double std) {
    const double z_alpha = 1.960;
    const double r = 5;

    double number = 0.0;
    double temp = 0.0;

    temp = (100 * z_alpha*std) / (r*mean);
    number = ceil(pow(temp, 2.0));

    return (int)number;

}

// This method is used for execute the matrix multiplication process for a given number of iterations for a given size of n
void test_bench(double **matA, double **matB, double **matTB, double **matC, double * execution_times, int n, int block_size, int num_execution_times)
{
    double dtime;
    for (size_t i = 0; i < num_execution_times; i++)
    {
        populate_array(matA, matB, n);

        dtime = omp_get_wtime();											// Start time
        get_parallel_matrix_mulT(matA, matB, matTB, matC, n, block_size);	// Execute matrix multiplication
        dtime = omp_get_wtime() - dtime;									// End time
        execution_times[i] = dtime;											// Record execution time

    }

}

//This method calculates the number of sufficient ammount of samples execution iterations need for the experiment
double calculate_sample_size(double **matA, double **matB, double **matTB, double **matC, int n, int block_size)
{
    int gen_iterations = 10;
    double  mean, std;
    double *execution_times;

    execution_times = new double[gen_iterations];
    test_bench(matA, matB, matTB, matC, execution_times, n, block_size, gen_iterations);
    mean = get_mean(execution_times, gen_iterations);
    std = get_standard_diviation(execution_times, mean, gen_iterations);

    return get_num_execution(mean, std);

}

// 2-Dimensional array deallocation
void free_memory(double ** data, int n)
{
    for (int i = 0; i < n; ++i) {
        delete[] data[i];
    }
}

int main() {
    int i, j, n, mat_size_min = 200, mat_size_max = 2000, steps = 200;
    int gen_iterations = 100;											// Initial sample size
    int suf_iterations = 0;												// Store the calculated sample size
    double **matA, **matB, **matC, **matBT, dtime, mean, std;
    double *execution_times;
    int block_size = 10;											// Define the block size for cache blocking

    for (size_t i = mat_size_min; i <= mat_size_max; i += steps)	// From 200 to 2000 in steps of 200
    {
        // Allocate memory for matrices

        matA = new double*[i];
        matB = new double*[i];
        matBT = new double*[i];
        matC = new double*[i];

        for (j = 0; j<i; j++) {

            matA[j] = new double[i];
            matB[j] = new double[i];
            matBT[j] = new double[i];
            matC[j] = new double[i];

        }

        /*suf_iterations = calculate_sample_size(mata, matb, matc, n);
        cout << "sufficient number of samples: " << suf_iterations;*/
        suf_iterations = 50;													// Sufficient amount of iterations
        execution_times = new double[suf_iterations];
        test_bench(matA, matB, matBT, matC, execution_times, i, block_size, suf_iterations);	// Execute the matrix multiplication for given number of iterations
        mean = get_mean(execution_times, suf_iterations);										// Calculate mean of execution times

        cout << "Mean execution time for matrix size " << i << "x" << i << " : " << mean << "s"<<endl;	// Print reults

        free_memory(matA, i);
        free_memory(matB, i);		// Free allocated memmory for 2-d arrays
        free_memory(matC, i);
        free_memory(matBT, i);

    }

    getchar();
    return 0;

}