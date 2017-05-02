#include <time.h>
#include <string.h>
#include "idw.h"

#define N_ITER 5

int main(int argc, char **argv)
{
    float   *zValues, 
            *zValuesGPU, 
            *knownValues, 
            *devZV,
            *devW, 
            *devKv;
    
    Point2D   *knownLocations, *queryLocations, *devQP, *devKP_blas;    
    int KN, QN, batchQN, batchDim, stride, MAX_SHMEM_SIZE, shMemSize, nIter, N_BATCH, type;     
    char *kp_filename, *loc_filename;
    
    // grid managing
    dim3 nBlocks, nThreadsForBlock;
    
    // gpu/cpu timing
    cudaEvent_t start, stop;
    float   cpuElapsedTime[N_ITER], cpuMeanTime, cpuSTD, 
            gpuElapsedTime[N_ITER], gpuMeanTime, gpuSTD;
    clock_t cpuStartTime;

    cudaDeviceProp prop;

    if (argc > 5)
    {
        type = atoi(argv[1]);
        batchQN = atoi(argv[4]);
        nThreadsForBlock.x = atoi(argv[5]);
    }
    else
    {
        printf("\nUsage:\n\n1) ./[bin_name] 1 [known_points_file] [query_locations_file] [QN_chunk_size] [block_threads_number]\n\n2) ./[bin_name] 2 [known_points_number] [query_locations_number] [QN_chunk_size] [block_threads_number]\n\n");
        exit(-1);
    }

    if (type == 1)
    {
        kp_filename = argv[2];
        loc_filename = argv[3];

        KN = getLines(kp_filename);
        QN = getLines(loc_filename);

        knownLocations = (Point2D *)malloc(KN*sizeof(Point2D));
        knownValues = (float *)malloc(KN*sizeof(float));
        queryLocations = (Point2D *)malloc(QN*sizeof(Point2D));

        generateDataset(kp_filename, knownLocations, knownValues);
        generateGrid(loc_filename, queryLocations);
    }
    else if (type == 2)
    {
        KN = atoi(argv[2]);
        QN = atoi(argv[3]);

        knownLocations = (Point2D *)malloc(KN*sizeof(Point2D));
        knownValues = (float *)malloc(KN*sizeof(float));
        queryLocations = (Point2D *)malloc(QN*sizeof(Point2D));

        // Random generation of 3D known points and 2D query points
        generateRandomData(knownLocations, knownValues, queryLocations, KN, QN);
    }

    /*for (int i=0; i<QN; i++)
        printf("%lf;%lf;\n",queryLocations[i].x,queryLocations[i].y);*/

    cudaGetDeviceProperties(&prop,0);
    MAX_SHMEM_SIZE = prop.sharedMemPerBlock/sizeof(Point2D);

    // known points are more than shared memory size?
    if (KN < MAX_SHMEM_SIZE)
    {
        shMemSize = KN*sizeof(Point2D);
        nIter = 1;
        stride = ceil((float)KN/(float)nThreadsForBlock.x);
    }
    else
    {
        shMemSize = MAX_SHMEM_SIZE*sizeof(Point2D);
        nIter = ceil((float)KN/(float)MAX_SHMEM_SIZE);
        stride = ceil((float)MAX_SHMEM_SIZE/(float)nThreadsForBlock.x);
    }
    
    zValues = (float*)malloc(QN*sizeof(float));
    zValuesGPU = (float*)malloc(QN*sizeof(float));

    if (batchQN > QN) batchQN = QN;
    nBlocks.x = ceil((float)batchQN/(float)nThreadsForBlock.x);
    N_BATCH = ceil((float)QN/(float)batchQN);

    cudaMalloc((void**)&devKP_blas, KN*sizeof(Point2D));
    cudaMalloc((void**)&devQP, batchQN*sizeof(Point2D));
    cudaMalloc((void**)&devZV, batchQN*sizeof(float));

    // device data for matrixVector
    cudaMalloc((void**)&devW, batchQN*KN*sizeof(float)); //weights matrix -> zquery = W*knownValues
    cudaMalloc((void**)&devKv, KN*sizeof(float)); //knownValues
    
    printf("Data generated!\n\n");

    printf("Number of known points: %d\n", KN);
    printf("Number of query points: %d\n", QN);
    printf("Iterations for query points: %d of max %d points\n", N_BATCH, batchQN);
    printf("Number of threads for block: %d\n", nThreadsForBlock.x);
    printf("Number of blocks: %d\n", nBlocks.x);
    printf("Shared memory size: %ld bytes\n", prop.sharedMemPerBlock);
    printf("Iterations for known points loading into shared memory: %d of max %d points\n", nIter, MAX_SHMEM_SIZE);
    printf("Stride: %d\n", stride);
    printf("Number of iterations for mean time generation: %d\n\n", N_ITER);

    /* --- CPU --- */
    cpuMeanTime = 0;
    for (int j=0; j<N_ITER; j++)
    {
        cpuStartTime = clock();
    
        sequentialIDW(knownLocations, knownValues, queryLocations, zValues, KN, QN);
    
        cpuElapsedTime[j] = ((float)(clock() - cpuStartTime))/CLOCKS_PER_SEC;
    
        printf("Elapsed CPU time : %f s\n" ,cpuElapsedTime[j]);

        cpuMeanTime += cpuElapsedTime[j];
    }
    
    cpuMeanTime /= N_ITER;
    cpuSTD = getSTD(cpuMeanTime, cpuElapsedTime, N_ITER);

    printf("Elapsed CPU MEAN time over %d iterations: %f s\n", N_ITER, cpuMeanTime);
    printf("CPU std: %f\n\n", cpuSTD);
    /* --- END CPU--- */

    /* -- GPU -- */

    gpuMeanTime = 0;
    for (int j=0; j<N_ITER; j++)
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start,0);

        batchDim = batchQN;
        cudaMemcpy(devKP_blas, knownLocations, KN*sizeof(Point2D), cudaMemcpyHostToDevice);
        cudaMemcpy(devKv, knownValues, KN*sizeof(float), cudaMemcpyHostToDevice);
        
        for (int i=0; i<N_BATCH; i++)
        {
            if (i == N_BATCH-1) batchDim = QN-i*batchQN;

            cudaMemcpy(devQP, &queryLocations[i*batchQN], batchDim*sizeof(Point2D), cudaMemcpyHostToDevice);

            parallelIDW<<<nBlocks,nThreadsForBlock,shMemSize>>>(devKP_blas, devQP, devW, devKv, devZV, KN, batchDim, stride, nIter, MAX_SHMEM_SIZE);

            cudaMemcpy(&zValuesGPU[i*batchQN], devZV, batchDim*sizeof(float), cudaMemcpyDeviceToHost);     
        }

        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuElapsedTime[j],start,stop);

        gpuElapsedTime[j] = gpuElapsedTime[j]*0.001;
        printf("Elapsed GPU time : %f s\n", gpuElapsedTime[j]);

        checkCUDAError("LAST ERROR:");

        gpuMeanTime += gpuElapsedTime[j];
    }

    /* --- END GPU --- */

    gpuMeanTime /= N_ITER;
    gpuSTD = getSTD(gpuMeanTime, gpuElapsedTime, N_ITER);

    printf("Elapsed GPU MEAN time over %d iterations : %f s\n", N_ITER, gpuMeanTime);
    printf("GPU std: %f\n", gpuSTD);

    if (updateLogCpuGpu(gpuMeanTime, cpuMeanTime, gpuSTD, cpuSTD, QN, KN, nBlocks.x, nThreadsForBlock.x) != -1) 
        printf("\nLog updated!\n");

    /*
    printf("Speed Up: %f\n\n", cpuElapsedTime/gpuElapsedTime);

    if (updateLog(gpuMeanTime, QN, KN, nBlocks.x, nThreadsForBlock.x) != -1) 
        printf("Log updated\n");
    */
    
    printf("Residue: %e\n", getRes(zValues, zValuesGPU, QN));

    if (saveData(knownLocations, KN, queryLocations, zValues, zValuesGPU, QN, cpuElapsedTime[0], gpuElapsedTime[0]) != -1)
        printf("Results saved! Look at your current directory! \n\n");

    free(knownLocations); free(queryLocations); free(knownValues); free(zValues); free(zValuesGPU);
    cudaFree(devKP_blas); cudaFree(devQP); cudaFree(devZV); cudaFree(devW); cudaFree(devKv);
    return 0;
}
