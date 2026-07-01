#include <iostream>
#include <vector>
#include <fstream>
#include <random>
#include <string>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA Error: " \
                      << cudaGetErrorString(err) \
                      << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define CHECK_CUBLAS(call) \
    do { \
        cublasStatus_t status = call; \
        if (status != CUBLAS_STATUS_SUCCESS) { \
            std::cerr << "cuBLAS Error" << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

void saveMatrix(const std::vector<double>& M,
                int rows,
                int cols,
                const std::string& filename)
{
    std::ofstream out(filename);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            out << M[i * cols + j];
            if (j < cols - 1)
                out << " ";
        }
        out << "\n";
    }
}

int main()
{
    const int N = 1000;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 10.0);

    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate(&handle));

    for (int pair = 1; pair <= 100; pair += 2) {

        std::vector<double> A(N * N);
        std::vector<double> B(N * N);
        std::vector<double> C(N * N);

        for (auto &x : A) x = dist(gen);
        for (auto &x : B) x = dist(gen);

        double *d_A, *d_B, *d_C;

        CHECK_CUDA(cudaMalloc(&d_A, N*N*sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_B, N*N*sizeof(double)));
        CHECK_CUDA(cudaMalloc(&d_C, N*N*sizeof(double)));

        CHECK_CUDA(cudaMemcpy(
            d_A, A.data(),
            N*N*sizeof(double),
            cudaMemcpyHostToDevice));

        CHECK_CUDA(cudaMemcpy(
            d_B, B.data(),
            N*N*sizeof(double),
            cudaMemcpyHostToDevice));

        const double alpha = 1.0;
        const double beta  = 0.0;

        CHECK_CUBLAS(
            cublasDgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N, N, N,
                &alpha,
                d_A, N,
                d_B, N,
                &beta,
                d_C, N
            )
        );

        CHECK_CUDA(cudaMemcpy(
            C.data(),
            d_C,
            N*N*sizeof(double),
            cudaMemcpyDeviceToHost));

        std::string filename =
            "pair" + std::to_string(pair) +
            "_pair" + std::to_string(pair + 1) +
            "_mult.txt";

        saveMatrix(C, N, N, filename);

        std::cout << "Saved " << filename << std::endl;

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    cublasDestroy(handle);

    return 0;
}
