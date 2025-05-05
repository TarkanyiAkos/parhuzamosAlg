#include "kernel_loader.h"
#include <stdio.h>
#include <stdlib.h>

//kernel forrás betöltése fájlból
char* loadKernelSource(const char *filename) { 
    FILE *fp = fopen(filename, "rb"); //read binary
    if (!fp) { //ha nem sikerült megnyitni
        fprintf(stderr, "Error: Failed to open kernel source file: %s\n", filename); //hibaüzenet
        return NULL; //null visszatérés hiba esetén
    }

    fseek(fp, 0, SEEK_END); //fájl végére ugrás
    size_t source_size = ftell(fp); //méret
    rewind(fp); //vissza a fájl elejére

	//memória foglalása a kernel kódnak
    char *source_str = (char*)malloc(source_size + 1); 
    if (!source_str) {
        fprintf(stderr, "Error: Failed to allocate memory for kernel source\n");
        fclose(fp);
        return NULL;
    }
	
	//fájl beolvasása a bufferbe
    fread(source_str, 1, source_size, fp);
	
	//null karakter a string végére
    source_str[source_size] = '\0'; 
    fclose(fp);
    return source_str;
}

//OpenCL program létrehozása és buildelése
cl_program createAndBuildProgram(cl_context context, cl_device_id device, const char *filename) { 
    char *source_str = loadKernelSource(filename);
	
    if (!source_str) { //hibaellenőrzés
        return NULL;
    }

    cl_int err; //hibakód változó
	
	
	//OpenCL program létrehozása a forrásból
    cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, NULL, &err); 
	
	//a forráskódhoz lefoglalt memória felszabadítása
    free(source_str); 

    if (err != CL_SUCCESS) {
        fprintf(stderr, "Error: Failed to create program with source (%d)\n", err);
        return NULL;
    }

	//program fordítása az adott eszközre
    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL); 
    if (err != CL_SUCCESS) { //ha fordítási hiba történt
	
        size_t log_size;
		
		//build log méretének lekérdezése
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size); 
		
		//memóriafoglalás a lognak
        char *log = (char *)malloc(log_size); 
		
		//build log beolvasása
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL); 
		
        fprintf(stderr, "Error in kernel build:\n%s\n", log); 
		
		//log memória felszabadítása
        free(log); 
		
		//program felszabadítása
        clReleaseProgram(program); 
		
		
        return NULL;
    }

    return program;
}
