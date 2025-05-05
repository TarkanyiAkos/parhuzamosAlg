#ifndef KERNEL_LOADER_H
#define KERNEL_LOADER_H

#include <CL/cl.h>

char* loadKernelSource(const char *filename); //kernel fkód betöltése

cl_program createAndBuildProgram(cl_context context, cl_device_id device, const char *filename); 
//OpenCL program létrehozása és fordítása a kernelből

#endif
