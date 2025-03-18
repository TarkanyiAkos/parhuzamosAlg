#include "kernel_loader.h"
#include <stdio.h>
#include <stdlib.h>

//kernel forrásbetöltés
char* loadKernelSource(const char *filename) {
    FILE *fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Failed to open kernel source file: %s\n", filename);
        return NULL;
    }
    //fájlméret meghatározása
    fseek(fp, 0, SEEK_END);
    size_t source_size = ftell(fp);
    rewind(fp);  //pointer visszaállítása

    //memória a forrásnak
    char *source_str = (char*)malloc(source_size + 1);
    if (!source_str) {
        fprintf(stderr, "Error: Failed to allocate memory for kernel source\n");
        fclose(fp);
        return NULL;
    }
    //fájlbeolvasás a kiadott bufferbe
    fread(source_str, 1, source_size, fp);
    //source_str[source_size] = ' ';
    source_str[source_size] = '\0';
    fclose(fp);
    return source_str;
}

//opencl program build a kernel forrásfájlokból
cl_program createAndBuildProgram(cl_context context, cl_device_id device, const char *filename) {
    //kernel forrásbetöltés
    char *source_str = loadKernelSource(filename);
    if (!source_str) {
        //
        return NULL;
    }
    cl_int err;
    //opencl program ekészítése
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, NULL, &err);
    free(source_str);  //forráskód memória felszabadítása
    if(err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to create program with source (%d)\n", err);
        return NULL;
    }
    //program compile adott eszközre
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if(err != CL_SUCCESS) {
        //ha nem sikerül lebuildelni, hiba
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        char *log = (char *)malloc(log_size);
        //build infó
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        fprintf(stderr, "Error in kernel build:\n%s\n", log);
        //log memória felszabadítás
        free(log);
        clReleaseProgram(program);
        return NULL;
    }
    return program;
}
