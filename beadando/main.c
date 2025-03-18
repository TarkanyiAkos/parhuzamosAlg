#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <CL/cl.h>
#include <math.h>
#include "kernel_loader.h"

#define NUM_WITNESSES 10 //Miller-Rabin tanúk száma

int main() {
    cl_int err;                           // opencl hiba hívások
    cl_platform_id platform_id = NULL;     // platform
    cl_uint num_platforms;                // elérhető platformok száma
    cl_device_id device_id = NULL;         // kiválasztott opencl eszköz tárolása
    cl_uint num_devices;                  // elérhető eszközök száma
    cl_context context = NULL;            // ocl kontextus eszközkezeléshez
    cl_command_queue command_queue = NULL;// parancs sor ütemezéshez
    cl_program program = NULL;            // kernel program
    cl_kernel kernel = NULL;              // kernel objektum függvény végrehajtáshoz

    //első elérhető opencl platform lekérésse
    err = clGetPlatformIDs(1, &platform_id, &num_platforms);
    if(err != CL_SUCCESS) {
        printf("Failed to get OpenCL platforn.\n");
        return 1;
    }

    //első elérhető opencl eszköz lekérése
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &num_devices);
    if(err != CL_SUCCESS) {
        printf("Failed to get OpenCL device.\n");
        return 1;
    }

    //Opencl kontextus létrehozása a kiválasztott eszközhöz
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
    if(err != CL_SUCCESS) {
        printf("Failed to create OpenCL context.\n");
        return 1;
    }

    //parancssor létrehozása művelet ütemezéshez
    command_queue = clCreateCommandQueue(context, device_id, 0, &err);
    if(err != CL_SUCCESS) {
        printf("Failed to create command queue.\n");
        return 1;
    }

    //opencl proram beolvasása és létrehozása (build)
    program = createAndBuildProgram(context, device_id, "sample.cl");
    if(program == NULL) {
        printf("Failed to create and build program.\n");
        return 1;
    }

    //kernel objektum létrehozása a buildelt programból, a millerRabinTest fügvényel
    kernel = clCreateKernel(program, "millerRabinTest", &err);
    if(err != CL_SUCCESS) {
        printf("Failed to create OpenCL kernel.\n");
        return 1;
    }

    //bit szám bekérése
    int n;
    printf("Enter the number of bits for the prime number: ");
    scanf("%d", &n);
    if(n < 3) {
        printf("ERROR! The given number is too small. Must be at least 3\n");
        return 1;
    }
    if(33 < n) {
        printf("ERROR! The given number is too large. Must be 33 or less.\n");
        return 1;
    }

    //alsóhatár: (2^(n-1)) felsőhatár: (2^n - 1) a prímhez
    unsigned long long lower_bound = 1ULL << (n - 1);
    unsigned long long upper_bound = (1ULL << n) - 1;





    srand(time(NULL));
    //prímjelölt létrehozása a megadott határokon belül
    unsigned long long candidate = lower_bound + rand() % (upper_bound - lower_bound + 1);
    //paritásteszt
    if(candidate % 2 == 0)
        candidate++;

    //loop amég nem prím
    int is_prime = 0;
    while(!is_prime) {
        //jelölt alatti páros szám első páratlan tényezőjénk előálítása
        unsigned long long d = candidate - 1;
        int s = 0;
        while((d % 2) == 0) {
            d /= 2;
            s++;
        }

        unsigned long long witnesses[NUM_WITNESSES]; // tanú array
        int results[NUM_WITNESSES];                  // kernel eredmények

        //random tanúk a megadott határokon belül, 2 és (candidate-2)
        for(int i = 0; i < NUM_WITNESSES; i++) {
            witnesses[i] = 2 + rand() % (candidate - 3);
        }



        //opencl buffer a jelölthöz
        cl_mem candidateBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                sizeof(unsigned long long), &candidate, &err);
        if(err != CL_SUCCESS) {
            printf("Failed to create buffer for candidate.\n");
            return 1;
        }
        //opencl buffer a tanúkhoz
        cl_mem witnessBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                              sizeof(unsigned long long) * NUM_WITNESSES, witnesses, &err);
        if(err != CL_SUCCESS) {
            printf("Failed to create buffer for witnesses.\n");
            return 1;
        }
        //buffer a kernel teszt eredményekhez
        cl_mem resultBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                             sizeof(int) * NUM_WITNESSES, NULL, &err);
        if(err != CL_SUCCESS) {
            printf("Failed to create buffer for results.\n");
            return 1;
        }

        //kernel argumentumok: candidate, witnesses, d, s, result buffer
        err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &candidateBuffer);
        if(err != CL_SUCCESS) {
            printf("Failed to set kernel argument 0.\n");
            return 1;
        }
        err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &witnessBuffer);
        if(err != CL_SUCCESS) {
            printf("Failed to set kernel argument 1.\n");
            return 1;
        }
        err = clSetKernelArg(kernel, 2, sizeof(unsigned long long), &d);
        if(err != CL_SUCCESS) {
            printf("Failed to set kernel argument 2.\n");
            return 1;
        }
        err = clSetKernelArg(kernel, 3, sizeof(int), &s);
        if(err != CL_SUCCESS) {
            printf("Failed to set kernel argument 3.\n");
            return 1;
        }
        err = clSetKernelArg(kernel, 4, sizeof(cl_mem), &resultBuffer);
        if(err != CL_SUCCESS) {
            printf("Failed to set kernel argument 4.\n");
            return 1;
        }




        //globáélis worksize
        size_t global_work_size = NUM_WITNESSES;
        //kernel parancsok sorbaállítása végrehajtásra az eszközön
        err = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL);
        if(err != CL_SUCCESS) {
            printf("Failed to enqueue kernel.\n");
            return 1;
        }
        //várakozás a kernel futásokra
        clFinish(command_queue);




        //eredmények kiolvasása a host arrayból
        err = clEnqueueReadBuffer(command_queue, resultBuffer, CL_TRUE, 0,
                                  sizeof(int) * NUM_WITNESSES, results, 0, NULL, NULL);
        if(err != CL_SUCCESS) {
            printf("Failed to read results from buffer.\n");
            return 1;
        }

        //a szám prím,h a bármelyik teszt 0, mégsem
        is_prime = 1;
        for(int i = 0; i < NUM_WITNESSES; i++) {
            if(results[i] == 0) {
                is_prime = 0;
                break;
            }
        }

        //buffer felszabadítás
        clReleaseMemObject(candidateBuffer);
        clReleaseMemObject(witnessBuffer);
        clReleaseMemObject(resultBuffer);

        //ha jelölt nem prím, következő jelölt
        if(!is_prime) {
            candidate += 2;
            if(candidate > upper_bound)
                candidate = lower_bound | 1ULL; //paritás!!!
        }
    }

    //első valós prím logolása
    printf("Found prime number: %llu\n", candidate);

    //felszabadítás
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    return 0;
}
