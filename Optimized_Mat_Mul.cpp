#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <random>

using namespace std;

inline void transpose(double **matA, double **matB, int n, int block_size) {

#pragma omp parallel
    {
        int_fast16_t i, j;
#pragma omp for
        for (i = 0; i < n; i += block_size) {
            for (j = 0; j < n; j+= block_size) {
                for (size_t k = 0; k < block_size; k++)
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

inline double get_vector_dot_product(double *vecA, double *vecB, int n) {

    double temp = 0.0;

    for (int_fast16_t i = 0; i < n / 8; i++) {
        temp += vecA[8 * i] * vecB[8 * i]
                + vecA[8 * i + 1] * vecB[8 * i + 1]
                + vecA[8 * i + 2] * vecB[8 * i + 2]
                + vecA[8 * i + 3] * vecB[8 * i + 3]
                + vecA[8 * i + 4] * vecB[8 * i + 4]
                + vecA[8 * i + 5] * vecB[8 * i + 5]
                + vecA[8 * i + 6] * vecB[8 * i + 6]
                + vecA[8 * i + 7] * vecB[8 * i + 7];
    }


    return temp;

}

//void get_mat_mulT(double **matA, double **matB, double **matC, int n)
//{
//	int_fast16_t i, j, k;
//	double **matB2;
//
//	matB2 = new double*[n];
//	for (i = 0; i<n; i++) {
//
//		matB2[i] = new double[n];
//
//	}
//
//	transpose(matB, matB2, n, blo);
//	for (i = 0; i < n; i++) {
//		for (j = 0; j < n; j++) {
//			double dot = 0;
//			for (k = 0; k < n; k++) {
//				dot += matA[i][k] * matB2[j][k];
//			}
//			matC[i][j] = dot;
//		}
//	}
//
//}

void get_parallel_matrix_mulT(double **matA, double **matB, double **matBT, double **matC, int n, int block_size)
{

    transpose(matB, matBT, n, block_size);
#pragma omp parallel
    {
        int_fast16_t i, j, k;
#pragma omp for
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                /*double dot = 0;
                for (k = 0; k < n; k++) {
                    dot += matA[i][k] * matB2[j][k];
                }*/
                matC[i][j] = get_vector_dot_product(matA[i], matBT[j], n);
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

// Run the test cases to collect sufficient number of samples and
void test_bench(double **matA, double **matB, double **matTB, double **matC, double * execution_times, int n, int block_size, int num_execution_times)
{
    double dtime;
    for (size_t i = 0; i < num_execution_times; i++)
    {
        populate_array(matA, matB, n);

        dtime = omp_get_wtime();
        get_parallel_matrix_mulT(matA, matB, matTB, matC, n, block_size);
        dtime = omp_get_wtime() - dtime;
        execution_times[i] = dtime;

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
    int i, j, n, mat_size_min = 200, mat_size_max = 1000, steps = 200;
    int gen_iterations = 100;
    int suf_iterations = 0;
    double **matA, **matB, **matC, **matBT, dtime, mean, std;
    double *execution_times;
    int block_size = 10;
    n = 1000;


    // Populate array

    for (size_t i = mat_size_min; i <= mat_size_max; i += steps)
    {
        // Allocate memory

        matA = new double*[i];
        matB = new double*[i];
        matBT = new double*[n];
        matC = new double*[i];

        for (j = 0; j<i; j++) {

            matA[j] = new double[i];
            matB[j] = new double[i];
            matBT[j] = new double[i];
            matC[j] = new double[i];

        }

        /*suf_iterations = calculate_sample_size(mata, matb, matc, n);
        cout << "sufficient number of samples: " << suf_iterations;*/
        suf_iterations = 5;
        execution_times = new double[suf_iterations];
        test_bench(matA, matB, matBT, matC, execution_times, i, block_size, suf_iterations);
        mean = get_mean(execution_times, suf_iterations);

        cout << "Mean execution time for matrix size " << i << "x" << i << " : " << mean << "s"<<endl;

        free_memory(matA, i);
        free_memory(matB, i);		// Free allocated memmory for 2-d arrays
        free_memory(matC, i);
        free_memory(matBT, i);

    }

    getchar();
    return 0;

}