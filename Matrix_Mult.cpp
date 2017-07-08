/* File:
 *     Matrix_Mult.cpp
 *
 *
 * Purpose:
 *     Computes a sequential matrix-matrix multiplication.
 *     Matrix size changes from 200 to 2000 in steps of 200.
 *
 * Input:
 *     None unless compiled with DEBUG flag.
 *
 * Output:
 *     Elapsed time for the computation for each size of matrix.
 *
 * Compile:
 *    g++ -fopenmp -lgomp -std=c++11 Matrix_Mult.cpp -o out
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


// Calculate the matrix multiplication of  matA and matB AXB
void get_matrix_mul(double **matA, double **matB, double **matC, int n)
{
    int i, j, k;
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            double mat_mul = 0;
            for (k = 0; k < n; k++) {
                mat_mul += matA[i][k] * matB[k][j];
            }
            matC[i][j] = mat_mul;
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

// Run the test cases to collect sufficient number of samples and
void test_bench(double **matA, double **matB, double **matC, double * execution_times, int n, int num_execution_times)
{
    double dtime;
    for (size_t i = 0; i < num_execution_times; i++)
    {
        populate_array(matA, matB, n);

        dtime = omp_get_wtime();
        get_matrix_mul(matA, matB, matC, n);
        dtime = omp_get_wtime() - dtime;
        execution_times[i] = dtime;

    }

}

//This method calculates the number of sufficient ammount of samples execution iterations need for the experiment
double calculate_sample_size(double **matA, double **matB, double **matC, int n)
{
    int gen_iterations = 10;
    double  mean, std;
    double *execution_times;

    execution_times = new double[gen_iterations];
    test_bench(matA, matB, matC, execution_times, n, gen_iterations);
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
    int i, j, n, mat_size_min = 200, mat_size_max = 2000, steps =200;
    int gen_iterations = 100;
    int suf_iterations = 0;
    double **matA, **matB, **matC, dtime, mean, std;
    double *execution_times;
    n = 200;


    // Populate array

    for (size_t i = mat_size_min; i <= mat_size_max; i += steps)
    {
        // Allocate memory

        matA = new double*[i];
        matB = new double*[i];
        matC = new double*[i];

        for (j = 0; j<i; j++) {

            matA[j] = new double[i];
            matB[j] = new double[i];
            matC[j] = new double[i];
        }


        /*suf_iterations = calculate_sample_size(mata, matb, matc, n);
        cout << "sufficient number of samples: " << suf_iterations;*/
        suf_iterations = 20;

        execution_times = new double[suf_iterations];
        test_bench(matA, matB, matC, execution_times, i, suf_iterations);
        mean = get_mean(execution_times, suf_iterations);

        cout << "Mean execution time for matrix size " << i << "x" << i << " : " << mean << "s" << endl;

        free_memory(matA, i);
        free_memory(matB, i);		// Free allocated memmory for 2-d arrays
        free_memory(matC, i);

    }

    getchar();
    return 0;

}