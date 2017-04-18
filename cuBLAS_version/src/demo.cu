#include <time.h>
#include <string.h>
#include "idw.h"

#define N_ITER 25 

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
            beta = 0;
    
    Point   *knownPoints;//, *devKP;	
    Point2D *queryPoints, *knownPoints_blas, *devQP, *devKP_blas;	
    int KN, QN, sizeKP, sizeKP_blas, sizeQP, stride, shMemSize, nIter; 	
    
    // grid managing
    dim3 nBlocks, nThreadsForBlock;
    
    // gpu/cpu timing
    cudaEvent_t start, stop;
    float   cpuElapsedTime[N_ITER], cpuMeanTime, cpuSTD, 
            gpuElapsedTime[N_ITER], gpuMeanTime, gpuSTD;
    clock_t cpuStartTime;

    cublasHandle_t handle;
    
    if (argc > 3)
    {
        KN = atoi(argv[1]);
        QN = atoi(argv[2]);
        nThreadsForBlock.x = atoi(argv[3]);
        nBlocks.x = ceil((float)QN/(float)nThreadsForBlock.x);
    }
    else
    {
        printf("\nUsage:\n\n ./[bin_name] [known_points_number] [locations_number] [block_threads_number]\n\n");
	    exit(-1);
    }
    
    sizeKP = KN*sizeof(Point);
    sizeKP_blas = KN*sizeof(Point2D);
    sizeQP = QN*sizeof(Point2D);
    
    // known points are more than shared memory size?
    if (KN < MAX_SHMEM_SIZE)
    {
        shMemSize = KN*sizeof(Point);
        nIter = 1;
        stride = ceil((float)KN/(float)nThreadsForBlock.x);
    }
    else
    {
        shMemSize = MAX_SHMEM_SIZE*sizeof(Point);
        nIter = ceil((float)KN/(float)MAX_SHMEM_SIZE);
        stride = ceil((float)MAX_SHMEM_SIZE/(float)nThreadsForBlock.x);
    }
    
    knownPoints = (Point*)malloc(sizeKP);
    knownPoints_blas = (Point2D*)malloc(sizeKP_blas);
    queryPoints = (Point2D*)malloc(sizeQP);
    knownValues = (float*)malloc(KN*sizeof(float)); // for cuBLAS
    zValues = (float*)malloc(QN*sizeof(float));
    zValuesGPU = (float*)malloc(QN*sizeof(float));
    W = (float*)malloc(QN*KN*sizeof(float));
    wSum = (float*)malloc(QN*sizeof(float));

    //cudaMalloc((void**)&devKP, sizeKP);
    cudaMalloc((void**)&devKP_blas, sizeKP_blas);
    cudaMalloc((void**)&devQP, sizeQP);
    cudaMalloc((void**)&devZV, QN*sizeof(float));

    // device data for cuBLAS
    cudaMalloc((void**)&devW, QN*KN*sizeof(float)); //weights matrix -> zquery = W*knownValues
    cudaMalloc((void**)&devKv, KN*sizeof(float)); //knownValues
    cudaMalloc((void**)&devWsum, QN*sizeof(float)); //wSum for each thread

    cublasCreate(&handle);

    // generating random data for testing
    generateRandomData(knownPoints, queryPoints, 0.0f, 100.0f, KN, QN);

    // adapting data for cuBLAS version
    for (int i=0; i<KN; i++)
    {
        knownValues[i] = knownPoints[i].z;
        knownPoints_blas[i].x = knownPoints[i].x;
        knownPoints_blas[i].y = knownPoints[i].y;
    }
    
    printf("Data generated!\n\n");

    printf("Number of known points: %d\n", KN);
    printf("Number of query points: %d\n", QN);
    printf("Number of threads for block: %d\n", nThreadsForBlock.x);
    printf("Number of blocks: %d\n", nBlocks.x);
    printf("Stride: %d\n", stride);
    printf("Number of iterations: %d of max %d points\n\n", nIter, MAX_SHMEM_SIZE);

    /* --- CPU --- */

    cpuMeanTime = 0;
    for (int j=0; j<N_ITER; j++)
    {
        cpuStartTime = clock();
    
        sequentialIDW(knownPoints, queryPoints, zValues, KN, QN);
    
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
    
        //cudaMemcpy(devKP, knownPoints, sizeKP, cudaMemcpyHostToDevice);
        cudaMemcpy(devKP_blas, knownPoints_blas, sizeKP_blas, cudaMemcpyHostToDevice);
        cudaMemcpy(devQP, queryPoints, sizeQP, cudaMemcpyHostToDevice);

        // Weights in devW
        computeWeights<<<nBlocks,nThreadsForBlock,shMemSize>>>(devKP_blas, devQP, devW, KN, QN, stride, devWsum, nIter); 

        cublasSetVector(KN, sizeof(float), knownValues, 1, devKv, 1);

        // Perform mat x vet using cublas (row-major)
        cublasSgemv(    handle,
                        CUBLAS_OP_T, 
                        KN, QN,
                        &alpha,
                        devW, KN,
                        devKv, 1,
                        &beta,
                        devZV, 1);

        // Complete weighted mean
        divideByWsum<<<nBlocks,nThreadsForBlock>>>(devZV, devWsum, QN); 

        cudaMemcpy(zValuesGPU, devZV, QN*sizeof(float), cudaMemcpyDeviceToHost);   

        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&gpuElapsedTime[j],start,stop);

        gpuElapsedTime[j] = gpuElapsedTime[j]*0.001;
        printf("Elapsed GPU time : %f s\n", gpuElapsedTime[j]);

        checkCUDAError("cublasSetVector");

        gpuMeanTime += gpuElapsedTime[j];
    }

    /* --- END GPU --- */

    gpuMeanTime /= N_ITER;
    gpuSTD = getSTD(gpuMeanTime, gpuElapsedTime, N_ITER);

    printf("Elapsed GPU MEAN time over %d iterations : %f s\n", N_ITER, gpuMeanTime);
    printf("GPU std: %f\n", gpuSTD);

    if (updateLogCpuGpu(gpuMeanTime, cpuMeanTime, gpuSTD, cpuSTD, QN, KN, nBlocks.x, nThreadsForBlock.x) != -1) 
        printf("\nLog updated\n");

    /*
    printf("Speed Up: %f\n\n", cpuElapsedTime/gpuElapsedTime);

    if (updateLog(gpuMeanTime, QN, KN, nBlocks.x, nThreadsForBlock.x) != -1) 
        printf("Log updated\n");
    */

    //getMaxAbsError(zValues, zValuesGPU, QN, &maxErr);
    //printf("Max error: %e\n",maxErr);
    printf("Residue: %e\n", getRes(zValues, zValuesGPU, QN));

    /*
    if (saveData(knownPoints, KN, queryPoints, zValues, zValuesGPU, QN, cpuElapsedTime[0], gpuElapsedTime[0]) != -1)
        printf("\nResults saved!\n");
    */

    free(knownPoints); free(knownPoints_blas); free(queryPoints); free(zValues); free(zValuesGPU); free(knownValues); free(W); free(wSum);
    /*cudaFree(devKP);*/ cudaFree(devKP_blas); cudaFree(devQP); cudaFree(devZV); cudaFree(devW); cudaFree(devKv); cudaFree(devWsum);

    cublasDestroy(handle);
     
    return 0;
}
