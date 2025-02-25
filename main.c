#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void add_vectors_sequential(float* vec1, float* vec2, float* result, size_t size);

void add_vectors_opencl(float* vec1, float* vec2, float* result, size_t size) {
    cl_int err;
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, 0, &err);

    cl_mem buffer_vec1 = clCreateBuffer(context, CL_MEM_READ_ONLY, size * sizeof(float), NULL, &err);
    cl_mem buffer_vec2 = clCreateBuffer(context, CL_MEM_READ_ONLY, size * sizeof(float), NULL, &err);
    cl_mem buffer_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(float), NULL, &err);

    clEnqueueWriteBuffer(queue, buffer_vec1, CL_TRUE, 0, size * sizeof(float), vec1, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, buffer_vec2, CL_TRUE, 0, size * sizeof(float), vec2, 0, NULL, NULL);

    const char* kernel_source = 
        "__kernel void add_vectors(__global const float* vec1, __global const float* vec2, __global float* result, const unsigned int size) {"
        "   int id = get_global_id(0);"
        "   if (id < size) {"
        "       result[id] = vec1[id] + vec2[id];"
        "   }"
        "}";

    cl_program program = clCreateProgramWithSource(context, 1, &kernel_source, NULL, &err);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "add_vectors", &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer_vec1);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &buffer_vec2);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &buffer_result);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &size);

    size_t global_work_size = size;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
    clFinish(queue);

    clEnqueueReadBuffer(queue, buffer_result, CL_TRUE, 0, size * sizeof(float), result, 0, NULL, NULL);

    clReleaseMemObject(buffer_vec1);
    clReleaseMemObject(buffer_vec2);
    clReleaseMemObject(buffer_result);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}

void add_vectors_sequential(float* vec1, float* vec2, float* result, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        result[i] = vec1[i] + vec2[i];
    }
}

int main() {
    srand(time(NULL));

    size_t N = 10; //dimenziószám
    float* vec1 = (float*)malloc(N * sizeof(float));
    float* vec2 = (float*)malloc(N * sizeof(float));
    float* result_sequential = (float*)malloc(N * sizeof(float));
    float* result_opencl = (float*)malloc(N * sizeof(float));
    for (size_t i = 0; i < N; ++i) {
        int r = rand() % 100;
        vec1[i] = r;
        r = rand() % 100;
        vec2[i] = r;
    }

    add_vectors_sequential(vec1, vec2, result_sequential, N);
    printf("Sequential:\n");
    for (size_t i = 0; i < N; ++i) {
        printf("result_sequential[%zu] = %.2f\n", i, result_sequential[i]);
    }

    add_vectors_opencl(vec1, vec2, result_opencl, N);
    printf("OpenCL:\n");
    for (size_t i = 0; i < N; ++i) {
        printf("result_opencl[%zu] = %.2f\n", i, result_opencl[i]);
    }

    int correct = 1;
    for (size_t i = 0; i < N; ++i) {
        if (result_sequential[i] != result_opencl[i]) {
            correct = 0;
            break;
        }
    }

    if (correct) {
        printf("Gud!\n");
    } else {
        printf("Not gud!\n");
    }

    free(vec1);
    free(vec2);
    free(result_sequential);
    free(result_opencl);

    return 0;
}
