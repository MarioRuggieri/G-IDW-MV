#ifndef idwheader
#define idwheader

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <cublas_v2.h>
#include <sys/stat.h>
#include <unistd.h>

#define PI 3.14159265
#define R 6371e3
//#define SEARCH_RADIUS 1000

struct point2D
{
    float x,y;
};

typedef struct point Point;
typedef struct point2D Point2D;

void checkCUDAError(const char* msg);

__device__ float havesineDistGPU(Point2D a, Point2D b);

__global__ void divideByWsum(float *devZV, float *devWsum, int QN); 

__global__ void computeWeights(   	Point2D *knownPoints, 
                                  	Point2D *queryPoints, 
                                    float *W, 
                                    int KN, 
                                    int QN, 
                                    int stride, 
                                    float* wSum,
                                    int nIter,
                                    int MAX_SHMEM_SIZE); 

float havesineDistCPU(Point2D a, Point2D b);

void sequentialIDW(Point2D *knownPoints, float* knownValues, Point2D *queryPoints, float *zValues, int KN, int QN);

void generateRandomData(Point2D *knownPoints, float *knownValues, Point2D *queryPoints, int KN, int QN);

int getLines(char *filename);

void generateGrid(char *filename, Point2D *queryLocations);

void generateDataset(char *filename, Point2D *knownLocations, float *knownValues);

int saveData(Point2D *knownPoints, int KN, Point2D *queryPoints, float *zValues, float *zValuesGPU, int QN, float cpuElapsedTime, float gpuElaspedTime);

int updateLog(float gpuMeanTime, int QN, int KN, int nBlocks, int nThreadsForBlock);

int updateLogCpuGpu(float gpuMeanTime, float cpuMeanTime, float gpuSTD, float cpuSTD, int QN, int KN, int nBlocks, int nThreadsForBlock);

void getMaxAbsError(float *zValues, float *zValuesGPU, int QN, float *maxErr);

float getRes(float *zValues, float *zValuesGPU, int QN);

float getSTD(float xm, float x[], int N);

#endif
