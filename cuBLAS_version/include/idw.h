#ifndef idwheader
#define idwheader

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <sys/stat.h>
#include <unistd.h>

#define MAX_SHMEM_SIZE 4096

struct point
{
    float x,y,z;
};

struct point2D
{
    float x,y;
};

typedef struct point Point;
typedef struct point2D Point2D;

void checkCUDAError(const char* msg);

__device__ float dist(Point2D a, Point b);

__global__ void divideByWsum(float *devZV, float *devWsum, int QN); 

__global__ void computeWeights(   	Point2D *knownPoints, 
                                  	Point2D *queryPoints, 
                                    float *W, 
                                    int KN, 
                                    int QN, 
                                    int stride, 
                                    float* wSum,
                                    int nIter); 

float cpuDist(Point2D a, Point b);

void sequentialIDW(Point *knownPoints, Point2D *queryPoints, float *zValues, int KN, int QN);

void generateRandomData(Point *knownPoints, Point2D *queryPoints, int a, int b, int N, int M);

int saveData(Point *knownPoints, int KN, Point2D *queryPoints, float *zValues, float *zValuesGPU, int QN, float cpuElapsedTime, float gpuElaspedTime);

int updateLog(float gpuMeanTime, int QN, int KN, int nBlocks, int nThreadsForBlock);

int updateLogCpuGpu(float gpuMeanTime, float cpuMeanTime, float gpuSTD, float cpuSTD, int QN, int KN, int nBlocks, int nThreadsForBlock);

void getMaxAbsError(float *zValues, float *zValuesGPU, int QN, float *maxErr);

float getRes(float *zValues, float *zValuesGPU, int QN);

float getSTD(float xm, float x[], int N);

void showData(Point *p, Point2D *pp, int N, int M);

#endif
