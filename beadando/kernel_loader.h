#ifndef KERNEL_LOADER_H
#define KERNEL_LOADER_H

#include <CL/cl.h>

//kernel forrásbetöltő
//return: dinamikusan allokált string, ami tartalmazza a forráskódot / NULL ha hiba
char* loadKernelSource(const char *filename);

//opencl program létrehozása a kernelből
//context, device, és filename a kernelből van olvasva
//return: buildelt cl_program / NULL ha hiba
cl_program createAndBuildProgram(cl_context context, cl_device_id device, const char *filename);

#endif
