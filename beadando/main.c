#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <CL/cl.h>
#include <math.h>
#include "kernel_loader.h"
#include <windows.h>

//Miller–Rabin tanúk száma
#define NUM_WITNESSES 100000
#define NUM_MEASUREMENTS 100

//időmérő függvény
double get_time_ms() { 
    static LARGE_INTEGER frequency; //frekvencia tárolása
    static int initialized = 0; //flag - inicializáltuk e már
    if (!initialized) { //ha még nem
        QueryPerformanceFrequency(&frequency); //órajel frekvencia lekérdezése
        initialized = 1; //flag: kész
    }
    LARGE_INTEGER now; //aktuális számláló
    QueryPerformanceCounter(&now); //számláló lekérdezése
    return (double)now.QuadPart * 1000.0 / frequency.QuadPart; //idő konvertálása ms-ra
}

int runTest(int bitCount) {
    cl_int err; //OpenCL hibakód
    cl_platform_id platform_id = NULL; //OpenCL platform azonosító
    cl_uint num_platforms; //elérhető platformok száma
    cl_device_id device_id = NULL; //OpenCL eszköz azonosító
    cl_uint num_devices; //elérhető eszközök száma
    cl_context context = NULL; //OpenCL kontextus
    cl_command_queue command_queue = NULL; //OpenCL parancssor
    cl_program program = NULL; //OpenCL program objektum
    cl_kernel kernel = NULL; //OpenCL kernel objektum

	//platformok lekérdezése
    err = clGetPlatformIDs(1, &platform_id, &num_platforms); 
    if(err != CL_SUCCESS) {
		//hiba
        printf("Failed to get OpenCL platform.\n");
        return 1;
    }
	
	//eszköz lekérdezése
    err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &num_devices);
    if(err != CL_SUCCESS) {
        printf("Failed to get OpenCL device.\n");
        return 1;
    }

	//kontextus létrehozása
    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err); 
    if(err != CL_SUCCESS) {
        printf("Failed to create OpenCL context.\n");
        return 1;
    }

	//parancssor létrehozása
    command_queue = clCreateCommandQueue(context, device_id, 0, &err); 
    if(err != CL_SUCCESS) {
        printf("Failed to create command queue.\n");
        return 1;
    }

	//kernel fájl betöltése és build
    program = createAndBuildProgram(context, device_id, "sample.cl"); 
    if(program == NULL) {
        printf("Failed to create and build program.\n");
        return 1;
    }

	//kernel objektum létrehozása
    kernel = clCreateKernel(program, "millerRabinTest", &err); 
    if(err != CL_SUCCESS) {
        printf("Failed to create OpenCL kernel.\n");
        return 1;
    }

    int n; //prím bitméret
	
    if (bitCount == 0) {
        printf("Enter the number of bits for the prime number: ");
        scanf("%d", &n);
        if(n < 3) {
            printf("Warning: Miller-Rabin algorythm cannot work with less than 3 bits, even though the following primes are only 2 bits long: 2 and 3.\n");
            return 1;
        }
        if(n > 32) {
            printf("ERROR! Max supported bits is 32.\n");
            return 1;
        }
    }
    else {
        n = bitCount;
    }

	//n-bites számok határai
    unsigned long long lower_bound = 1ULL << (n - 1);
    unsigned long long upper_bound = (1ULL << n) - 1;

    srand(time(NULL)); //véletlenszám generátor inic

	//PÁRHUZAMOS
    if (bitCount == 0) {
        printf("\n--- Running OpenCL primality test ---\n");
    }
    double start = get_time_ms(); //időmérés kezdése

    unsigned long long candidate = lower_bound + rand() % (upper_bound - lower_bound + 1); //kezdeti prímjelölt
    unsigned long long start_candidate = candidate; //elmentjük a kiinduló értéket
    if(candidate % 2 == 0)
        candidate++; //csak páratlan számokat vizsgálunk

    int is_prime = 0; //kezdetben még nem prím
    while(!is_prime) { //amíg nem találunk prímet
        unsigned long long d = candidate - 1;
        int s = 0;
        while((d % 2) == 0) {
            d /= 2;
            s++;
        }

        unsigned long long witnesses[NUM_WITNESSES]; //tanúk tömbje
        int results[NUM_WITNESSES]; //tesztek eredményei

        for(int i = 0; i < NUM_WITNESSES; i++)
            witnesses[i] = 2 + rand() % (candidate - 3); //véletlenszerű tanúk generálása

        cl_mem candidateBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned long long), &candidate, &err); //jelölt buffer
        cl_mem witnessBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(unsigned long long) * NUM_WITNESSES, witnesses, &err); //tanúk buffer
        cl_mem resultBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * NUM_WITNESSES, NULL, &err); //eredmények buffer

        clSetKernelArg(kernel, 0, sizeof(cl_mem), &candidateBuffer); //kernel argumentum beállítás
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &witnessBuffer);
        clSetKernelArg(kernel, 2, sizeof(unsigned long long), &d);
        clSetKernelArg(kernel, 3, sizeof(int), &s);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &resultBuffer);

        size_t global_work_size = NUM_WITNESSES; //globális méret beállítása
        clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_work_size, NULL, 0, NULL, NULL); //kernel futtatása
        clFinish(command_queue); //várakozás a futás végére

        clEnqueueReadBuffer(command_queue, resultBuffer, CL_TRUE, 0, sizeof(int) * NUM_WITNESSES, results, 0, NULL, NULL); //eredmények kiolvasása

        is_prime = 1; //kezdetben feltételezzük hogy prím
        for(int i = 0; i < NUM_WITNESSES; i++) {
            if(results[i] == 0) {
                is_prime = 0; //ha bármelyik tanú szerint nem prím
                break;
            }
        }

        clReleaseMemObject(candidateBuffer); //memóriák felszabadítása
        clReleaseMemObject(witnessBuffer);
        clReleaseMemObject(resultBuffer);

        if(!is_prime) {
            candidate += 2; //ugrás a következő páratlan számra
            if(candidate > upper_bound)
                candidate = lower_bound | 1ULL; //vissza a tartomány elejére
        }
    }


    double end = get_time_ms(); //időmérés vége
	

    if (bitCount == 0) {
        printf("OpenCL found prime: %llu\n", candidate); //talált prím kiírása
        printf("Time taken (OpenCL): %.3f ms\n", end - start); //idő kiírása

        printf("\n--- Running Sequential primality test ---\n");
    }
	float parTime = end - start;
    //SZEKVENCIÁLIS
    
    start = get_time_ms(); //időmérés kezdete

    candidate = start_candidate; //eredeti jelölt
    if(candidate % 2 == 0)
        candidate++;

    while(1) { //amíg nem találunk prímet
        unsigned long long d = candidate - 1;
        int s = 0;
        while((d % 2) == 0) {
            d /= 2;
            s++;
        }

        int is_prime_seq = 1; //prím állapot
        for(int i = 0; i < NUM_WITNESSES; i++) {
            unsigned long long a = 2 + rand() % (candidate - 3); //véletlen tanú
            unsigned long long x = 1;
            unsigned long long base = a;
            unsigned long long exp = d;

            base %= candidate;
            while(exp > 0) {
                if(exp & 1)
                    x = (x * base) % candidate;
                exp >>= 1;
                base = (base * base) % candidate;
            }

            if(x == 1 || x == candidate - 1)
                continue;

            int continueLoop = 0;
            for(int r = 1; r < s; r++) {
                x = (x * x) % candidate;
                if(x == candidate - 1) {
                    continueLoop = 1;
                    break;
                }
                if(x == 1) {
                    break;
                }
            }
            if(continueLoop)
                continue;

            is_prime_seq = 0;
            break;
        }

        if(is_prime_seq)
            break;

        candidate += 2;
        if(candidate > upper_bound)
            candidate = lower_bound | 1ULL;
    }

    end = get_time_ms();
    if (bitCount == 0) {
        printf("Sequential found prime: %llu\n", candidate);
        printf("Time taken (Sequential): %.3f ms\n", end - start);
    }
    float seqTime = end - start;
    clReleaseKernel(kernel); //OpenCL erőforrások felszabadítása
    clReleaseProgram(program);
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);

    if (bitCount != 0) {
        FILE *fp;
        //log bitszám
        fp=fopen("data_bitcounts.txt", "a");
        if(fp == NULL) exit(-1); //hiba
        fprintf(fp, "%d\n", n);
        fclose(fp);

        //log parhuzamos
        fp=fopen("data_par.txt", "a");
        if(fp == NULL) exit(-1); //hiba
        fprintf(fp, "%.3f\n", parTime);
        fclose(fp);
        
        //log szekvenciális
        fp=fopen("data_seq.txt", "a");
        if(fp == NULL) exit(-1); //hiba
        fprintf(fp, "%.3f\n", seqTime);
        fclose(fp);
    }

    return 0;
}

int main() {
    
    printf("--------------------------\nPrime Number Generator\n\n1) Manual Mode\n2) Automated Mode\n3) Exit Program\n\nSelect option (1-3): ");
    int answ = 0;
    scanf("%d", &answ);
    if (answ == 1) {
        runTest(0);
    }
    if (answ == 3) {
        printf("Program closed.");
    }
    if (answ == 2) {
        //takaritas
        remove("data_bitcounts.txt");
        remove("data_par.txt");
        remove("data_seq.txt");
        for (int i = 0; i < 30; i++) {
            for (int j = 0; j < NUM_MEASUREMENTS; j++) {
                runTest(i+3);
                printProgressBar(i * NUM_MEASUREMENTS + j, 30 * NUM_MEASUREMENTS);
            }
        }
        printProgressBar(1, 1);
    }
    
    return 0;
}

void printProgressBar(int progress, int total) {
    int barWidth = 100;
    float percentage = (float)progress / total * 100;

    printf("\r[");
    int pos = barWidth * progress / total;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) {
            printf("=");
        } else if (i == pos) {
            printf(">");
        } else {
            printf(" ");
        }
    }
    printf("] %.2f%%", percentage);
    fflush(stdout);
}