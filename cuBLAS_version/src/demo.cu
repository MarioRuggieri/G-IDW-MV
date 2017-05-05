#include <time.h>
#include <string.h>
#include "idw.h"

int main(int argc, char **argv)
{
    float   *zValues, 
            *zValuesGPU, 
            *knownValues, 
            *wSum,
            *devZV,
            *devW, 
            *W,
            *devKv,
            *devWsum,
            alpha = 1,
            beta = 0,
            searchRadius;
    
    Point2D   *knownLocations, *queryLocations, *devQP, *devKP_blas;	
    int KN, QN, batchQN, batchDim, stride, MAX_SHMEM_SIZE, shMemSize, nIter, N_BATCH, type; 	
    char *kp_filename, *loc_filename;
    
    // grid managing
    dim3 nBlocks, nThreadsForBlock;
    
    // gpu/cpu timing
    cudaEvent_t start, stop;
    float   cpuElapsedTime, 
            gpuElapsedTime;
    clock_t cpuStartTime;

    cublasHandle_t handle;
    cudaDeviceProp prop;

    if (argc > 6)
    {
        type = atoi(argv[1]);
        batchQN = atoi(argv[4]);
        nThreadsForBlock.x = atoi(argv[5]);
        searchRadius = atoi(argv[6]);
    }
    else
    {
        printf("\nUsage:\n\n1) ./[bin_name] 1 [known_points_file] [query_locations_file] [QN_chunk_size] [block_threads_number] [search_radius]\n\n2) ./[bin_name] 2 [known_points_number] [query_locations_number] [QN_chunk_size] [block_threads_number] [search_radius]\n\n");
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
    else 
    {
        printf("Type must be 1 or 2!");
        exit(-1);
    }

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
    W = (float*)malloc(QN*KN*sizeof(float));
    wSum = (float*)malloc(QN*sizeof(float));
    memset(zValuesGPU,0,QN*sizeof(float));

    if (batchQN > QN) batchQN = QN;

    nBlocks.x = ceil((float)batchQN/(float)nThreadsForBlock.x);
    N_BATCH = ceil((float)QN/(float)batchQN);

    cudaMalloc((void**)&devKP_blas, KN*sizeof(Point2D));
    cudaMalloc((void**)&devQP, batchQN*sizeof(Point2D));
    cudaMalloc((void**)&devZV, batchQN*sizeof(float));

    // device data for cuBLAS
    cudaMalloc((void**)&devW, batchQN*KN*sizeof(float)); //weights matrix -> zquery = W*knownValues
    cudaMalloc((void**)&devKv, KN*sizeof(float)); //knownValues
    cudaMalloc((void**)&devWsum, batchQN*sizeof(float)); //wSum for each thread

    cublasCreate(&handle);
    
    printf("Data generated!\n\n");

    printf("Number of known points: %d\n", KN);
    printf("Number of query points: %d\n", QN);
    printf("Iterations for query points: %d of max %d points\n", N_BATCH, batchQN);
    printf("Number of threads for block: %d\n", nThreadsForBlock.x);
    printf("Number of blocks: %d\n", nBlocks.x);
    printf("Shared memory size: %ld bytes\n", prop.sharedMemPerBlock);
    printf("Iterations for known points loading into shared memory: %d of max %d points\n", nIter, MAX_SHMEM_SIZE);
    printf("Stride: %d\n\n", stride);

    /* --- CPU --- */
    printf("Executing on CPU...\n");

    cpuStartTime = clock();
    
    if (sequentialIDW(knownLocations, knownValues, queryLocations, zValues, KN, QN, searchRadius) < 0)
    {
        printf("Search radius is too small! Some values cannot be interpolated!\nYou need more dataset points or a different search radius!\n");
        exit(-1);
    }
    
    cpuElapsedTime = ((float)(clock() - cpuStartTime))/CLOCKS_PER_SEC;
    
    printf("Elapsed CPU time : %f s\n" ,cpuElapsedTime);    
    /* --- END CPU--- */

    /* -- GPU -- */
    printf("\nExecuting on GPU...\n");

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start,0);

    batchDim = batchQN;
    cudaMemcpy(devKP_blas, knownLocations, KN*sizeof(Point2D), cudaMemcpyHostToDevice);
    cublasSetVector(KN, sizeof(float), knownValues, 1, devKv, 1);
    cublasSetVector(batchQN, sizeof(float), zValuesGPU, 1, devZV, 1);

    for (int i=0; i<N_BATCH; i++)
    {
        if (i == N_BATCH-1) batchDim = QN-i*batchQN;

        cudaMemcpy(devQP, &queryLocations[i*batchQN], batchDim*sizeof(Point2D), cudaMemcpyHostToDevice);

        // Weights in devW
        computeWeights<<<nBlocks,nThreadsForBlock,shMemSize>>>(devKP_blas, devQP, devW, KN, batchDim, stride, devWsum, nIter, MAX_SHMEM_SIZE, searchRadius); 

        // Perform mat x vet using cublas (row-major)
        cublasSgemv(    handle,
                        CUBLAS_OP_T, 
                        KN, batchDim,
                        &alpha,
                        devW, KN,
                        devKv, 1,
                        &beta,
                        devZV, 1);

        // Complete weighted mean
        divideByWsum<<<nBlocks,nThreadsForBlock>>>(devZV, devWsum, batchDim); 

        cudaMemcpy(&zValuesGPU[i*batchQN], devZV, batchDim*sizeof(float), cudaMemcpyDeviceToHost);   
    }

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpuElapsedTime,start,stop);

    gpuElapsedTime = gpuElapsedTime*0.001;
    printf("Elapsed GPU time : %f s\n", gpuElapsedTime);

    checkCUDAError("LAST ERROR:");
    /* --- END GPU --- */

    if (updateLogCpuGpu(gpuElapsedTime, cpuElapsedTime, QN, KN, nBlocks.x, nThreadsForBlock.x) != -1) 
        printf("\nLog updated!\n");

    /*
    printf("Speed Up: %f\n\n", cpuElapsedTime/gpuElapsedTime);

    if (updateLog(gpuMeanTime, QN, KN, nBlocks.x, nThreadsForBlock.x) != -1) 
        printf("Log updated\n");
    */
	
    printf("Residue: %e\n", getRes(zValues, zValuesGPU, QN));

    if (saveData(knownLocations, KN, queryLocations, zValues, zValuesGPU, QN, cpuElapsedTime, gpuElapsedTime) != -1)
        printf("Results saved! Look at your current directory! \n\n");
    
    free(knownLocations); free(queryLocations); free(knownValues); free(zValues); free(zValuesGPU); free(W); free(wSum);
    cudaFree(devKP_blas); cudaFree(devQP); cudaFree(devZV); cudaFree(devW); cudaFree(devKv); cudaFree(devWsum);

    cublasDestroy(handle);

    return 0;
}
