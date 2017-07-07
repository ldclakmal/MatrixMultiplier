#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <random>

using namespace std;


// calculate the matrix multiplication of  mata and matb using loop parallel 
void get_parallel_matrix_mul(double **mata, double **matb, double **matc, int n)
{
#pragma omp parallel		// Define the parallel code block 
    {
        int i, j, k;
#pragma omp for             // Make the outer loop parallel 
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                double mat_mul = 0;
                for (k = 0; k < n; k++) {
                    mat_mul += mata[i][k] * matb[k][j];
                }
                matc[i][j] = mat_mul;
            }
        }

    }
}

// return a random double number between given two double values 
double frand(double fmin, double fmax)
{
    random_device rd;										// seed
    mt19937 mt(rd());										// random number generator
    uniform_real_distribution<double> dist(fmin, fmax);		// distribution of numbers

    return dist(mt);
}

// populate given 2-d arrays using random double values 
void populate_array(double **mata, double **matb, int n)
{
    int i = 0, j;

    for (i = 0; i<n; i++) {
        for (j = 0; j < n; j++)
        {
            mata[i][j] = frand(1.0, 2000.0);
            matb[i][j] = frand(1.0, 2000.0);
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

//returns the mean of set of floats
double get_mean(double *execution_times, int num_execution_times) {
    double sum = 0.0f;
    for (int i = 0; i < num_execution_times; i++) {
        sum += execution_times[i];
    }
    return sum / (double)num_execution_times;
}

//returns the standard deviation of set of floats
double get_standard_diviation(double *execution_times, double mean, int num_execution_times) {
    double sum = 0.0f;
    for (int i = 0; i < num_execution_times; i++) {
        sum += (execution_times[i] - mean)*(execution_times[i] - mean);
    }
    return (double)sqrt(sum / (double)(num_execution_times - 1));
}

//returns required number of executions, provided mean and standard deviation 
int get_num_execution(double mean, double std) {
    const double z_alpha = 1.960;
    const double r = 5;

    double number = 0.0;
    double temp = 0.0;

    temp = (100 * z_alpha*std) / (r*mean);
    number = ceil(pow(temp, 2.0));

    return (int)number;

}

// run the test cases to collect sufficient number of samples and 
void test_bench(double **mata, double **matb, double **matc, double * execution_times, int n, int num_execution_times)
{
    double dtime;
    for (size_t i = 0; i < num_execution_times; i++)
    {
        populate_array(mata, matb, n);

        dtime = omp_get_wtime();
        get_parallel_matrix_mul(mata, matb, matc, n);
        dtime = omp_get_wtime() - dtime;
        execution_times[i] = dtime;

    }

}

//This method calculates the number of sufficient ammount of samples execution iterations need for the experiment 

double calculate_sample_size(double **mata, double **matb, double **matc, int n)
{
    int gen_iterations = 10;
    double  mean, std;
    double *execution_times;

    execution_times = new double[gen_iterations];
    test_bench(mata, matb, matc, execution_times, n, gen_iterations);
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



    // populate array

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