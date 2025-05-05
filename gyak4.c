#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>

#define MATRIX_SIZE 5

const char* kernelSource = R"(
__kernel void matrix_multiply(__global const float* A, 
                              __global const float* B, 
                              __global float* C, 
                              const unsigned int N) {
    int row = get_global_id(1);
    int col = get_global_id(0);
    
    float sum = 0;
    for (int i = 0; i < N; ++i) {
        sum += A[row * N + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}
)";

void initialize_matrix(float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            matrix[i * size + j] = rand() % 100;
        }
    }
}

void print_matrix(float* matrix, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            printf("%.2f ", matrix[i * size + j]);
        }
        printf("\n");
    }
}

int main() {
    
    int N = MATRIX_SIZE;
    
    float* A = (float*)malloc(sizeof(float) * N * N);
    float* B = (float*)malloc(sizeof(float) * N * N);
    float* C = (float*)malloc(sizeof(float) * N * N);
    
    initialize_matrix(A, N);
    initialize_matrix(B, N);
    printf("\nMatrix A:\n");
    print_matrix(A, N);
    printf("\nMatrix B:\n");
    print_matrix(B, N);

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * N * N, NULL, NULL);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * N * N, NULL, NULL);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * N * N, NULL, NULL);

    clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, sizeof(float) * N * N, A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, sizeof(float) * N * N, B, 0, NULL, NULL);

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "matrix_multiply", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &N);
    size_t global_work_size[2] = {N, N};
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, NULL, 0, NULL, NULL);
    clFinish(queue);
    clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, sizeof(float) * N * N, C, 0, NULL, NULL);
    printf("\n----------\n\nResult matrix C:\n");
    print_matrix(C, N);

    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(A);
    free(B);
    free(C);
    return 0;
}
