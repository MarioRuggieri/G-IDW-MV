#include "idw.h"

void checkCUDAError(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Cuda error: %s %s\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}

__device__ float havesineDistGPU(Point2D p1, Point2D p2)
{
    float   lat1 = PI*p1.y/180,
            lat2 = PI*p2.y/180,
            dlat = PI*(p2.y-p1.y)/180,
            dlon = PI*(p2.x-p1.x)/180,
            a = sin(dlat/2) * sin(dlat/2) + cos(lat1) * cos(lat2) * sin(dlon/2) * sin(dlon/2),
            c = 2 * atan2(sqrt(a), sqrt(1-a));

    return R * c;
}

__global__ void divideByWsum(float *zValues, float *wSum, int QN)
{
    int ind = threadIdx.x + blockIdx.x*blockDim.x;

    if (ind < QN) 
        zValues[ind] /= wSum[ind];
}

// IDW parallel GPU version
__global__ void computeWeights(     Point2D *knownPoints, 
                                    Point2D *queryPoints, 
                                    float *W, 
                                    int KN, 
                                    int QN, 
                                    int stride,
                                    float *wSum,
                                    int nIter,
                                    int MAX_SHMEM_SIZE)
{
    extern __shared__ Point2D shMem[];
    int ind = threadIdx.x + blockIdx.x*blockDim.x, smStartInd, startInd, i, k, currentKN, shift, work = 1;
    float my_wSum = 0, w, d;
    Point2D myPoint, p;
    
    shift = 0;
    currentKN = MAX_SHMEM_SIZE;	//chunk current dimension

    // each iteration fills as much as possible shared memory
    for (k = 0; k < nIter; k++)
    {
        //the last or only one iteration
        if (currentKN > KN) currentKN = KN;
        
        /* --- loading known points into shared memory --- */
        
        smStartInd = threadIdx.x*stride;

        //shift used to move into knownPoints array for chunk selection
        startInd = smStartInd + shift;  
        //if (ind == 0) printf("startInd: %d\n",startInd);

        if (startInd < currentKN) 
        {
            i = 0;
            while (i < stride && (startInd + i) < currentKN) // for the last thread: <= stride points
            {
                shMem[smStartInd + i] = knownPoints[startInd + i];
                i++;
            }
        }

        __syncthreads();

        /* --- loading finished --- */
        
        if (work)
        {
            // updating the interpolated z value for each thread
            if (ind < QN) 
            {
                myPoint = queryPoints[ind]; // some block threads are not used

                for (i = 0; i < currentKN-shift; i++)
                {
                    p = shMem[i];

                    d = havesineDistGPU(myPoint,p);
                    //d = sqrt((myPoint.x - p.x)*(myPoint.x - p.x) + (myPoint.y - p.y)*(myPoint.y - p.y));

                    if (d != 0)
                    {
                        //if (d < SEARCH_RADIUS)
                        //{
                        //if (ind == 0) printf("%d\n",ind*KN + i+k*MAX_SHMEM_SIZE);
                            w = 1/(d*d);
                            W[ind*KN + i+k*MAX_SHMEM_SIZE] = w;
                            my_wSum += w;
                        /*}
                        else
                        {
                            W[ind*KN + i+k*MAX_SHMEM_SIZE] = 0;
                        }*/
                    }
                    else
                    {
                        for (int l=0; l<KN; l++) W[ind*KN + l] = 0;
                        W[ind*KN + i+k*MAX_SHMEM_SIZE] = 1; //1 for the zero distance point
                        my_wSum = 1;
                        work = 0;
                        break; 
                    }
                }
            }       
        }
	    
	shift = currentKN;
        currentKN += MAX_SHMEM_SIZE; 

        __syncthreads();
    }

    if (ind < QN)
    {
        wSum[ind] = my_wSum;
    }
    
}

float havesineDistCPU(Point2D p1, Point2D p2)
{
    float   lat1 = PI*p1.y/180,
            lat2 = PI*p2.y/180,
            dlat = PI*(p2.y-p1.y)/180,
            dlon = PI*(p2.x-p1.x)/180,
            a = sin(dlat/2) * sin(dlat/2) + cos(lat1) * cos(lat2) * sin(dlon/2) * sin(dlon/2),
            c = 2 * atan2(sqrt(a), sqrt(1-a));

    return R * c;
}

// IDW sequential CPU version
void sequentialIDW(Point2D *knownPoints, float* knownValues, Point2D *queryPoints, float *zValues, int KN, int QN)
{
    int i,j;
    float wSum, w, d;
    
    for (i=0; i<QN; i++)
    {
        wSum = 0; zValues[i] = 0;

        for (j=0; j<KN; j++)
        {
            d = havesineDistCPU(queryPoints[i],knownPoints[j]);
            /*d = sqrt( (queryPoints[i].x - knownPoints[j].x)*(queryPoints[i].x - knownPoints[j].x) + 
                        (queryPoints[i].y - knownPoints[j].y)*(queryPoints[i].y - knownPoints[j].y));*/

            if (d != 0)
            {
                //if (d < SEARCH_RADIUS)
                //{
                    w = 1/(d*d);
                    wSum += w;
                    zValues[i] += w*knownValues[j];
                //}
            }
            else
            {
                zValues[i] = knownValues[j];
		        wSum = 1;
                break;
            }
        }
        
        zValues[i] /= wSum;
    }
}

// Random generation of 2D known points and 2D query points
void generateRandomData(Point2D *knownPoints, float *knownValues, Point2D *queryPoints, int KN, int QN)
{
    int i;
    srand((unsigned int)time(NULL));

    for (i=0; i<KN; i++)
    {
        knownPoints[i].x = (rand()/(float)(RAND_MAX))* 180;
        knownPoints[i].y = (rand()/(float)(RAND_MAX))* 180;
        knownValues[i] = (rand()/(float)(RAND_MAX))* 180;
    }

    for (i=0; i<QN; i++)
    {
        queryPoints[i].x = (rand()/(float)(RAND_MAX))* 90;
        queryPoints[i].y = (rand()/(float)(RAND_MAX))* 90;
    }
    
}

int getLines(char *filename)
{
    FILE *fp = fopen(filename,"r");
    int ch = 0, cont = 1;

    while(!feof(fp))
    {
        ch = fgetc(fp);
        if(ch == '\n')
        {
            cont++;
        }
    }

    fclose(fp);

    return cont;
}


void generateDataset(char *filename, Point2D *knownLocations, float *knownValues)
{
    FILE *fp = fopen(filename,"r");
    int i=0;

    while(fscanf(fp,"%f;%f;%f;",&knownLocations[i].x,&knownLocations[i].y,&knownValues[i]) == 3)
    {
        i++;
    }

    fclose(fp);
}

void generateGrid(char *filename, Point2D *queryLocations)
{
    FILE *fp = fopen(filename,"r");
    int i=0;

    while(fscanf(fp,"%f;%f;",&queryLocations[i].x,&queryLocations[i].y) == 2 )
    {
        i++;
    }

    fclose(fp);
}


int saveData(Point2D *knownPoints, int KN, Point2D *queryPoints, float *zValues, float *zValuesGPU, int QN, float cpuElapsedTime, float gpuElaspedTime)
{
    FILE *f;
    time_t t;
    struct tm *tm;
    char *directory, date[30], *myDir;

    t = time(NULL);
    tm = localtime(&t);
    strftime(date, sizeof(date)-1, "%d-%m-%Y_%H:%M:%S", tm);
    directory = "Results-";

    myDir = (char *)malloc(strlen(directory)+strlen(date)+1);
    strcpy(myDir, directory);
    strcat(myDir, date);

    if( mkdir(myDir,0777) < 0 ) 
    {
       printf("Cannot create directory\n");
       return(-1);   
    }

    if (chdir(myDir) < 0)
    {
        printf("Cannot change directory\n");
        return(-1);
    }

    // Saving generated data
    /*
    f = fopen("generatedData.txt", "w");
    if (f == NULL)
    {
        printf("Error opening generatedData file!\n");
        return(-1);
    }
    
    for (int i=0; i<KN; i++)
        fprintf(f, "(x: %f, y: %f, z: %f)\n", knownPoints[i].x, knownPoints[i].y, knownPoints[i].z);
    
    fclose(f);
    */
    // Saving CPU output
    f = fopen("cpuOutput.txt", "w");
    if (f == NULL)
    {
        printf("Error opening cpuOutput file!\n");
        return(-1);
    }
    
    for (int i=0; i<QN; i++)
        fprintf(f, "(x: %f, y: %f, z: %f)\n", queryPoints[i].x, queryPoints[i].y, zValues[i]);
    
    fclose(f);

    // Saving GPU output
    f = fopen("gpuOutput.txt", "w");
    if (f == NULL)
    {
        printf("Error opening gpuOutput file!\n");
        return(-1);
    }
    
    for (int i=0; i<QN; i++)
        fprintf(f, "(x: %f, y: %f, z: %f)\n", queryPoints[i].x, queryPoints[i].y, zValuesGPU[i]);
    
    fclose(f);

    // Saving times
    f = fopen("times.txt", "w");
    if (f == NULL)
    {
        printf("Error opening times file!\n");
        return(-1);
    }
    
    fprintf(f, "Cpu Elapsed Time: %f\n Gpu Elasped Time: %f\n Speed Up: %f", 
                cpuElapsedTime, gpuElaspedTime, cpuElapsedTime/gpuElaspedTime);
    
    fclose(f);

    return 0;
}

int updateLog(float gpuMeanTime, int QN, int KN, int nBlocks, int nThreadsForBlock)
{
    FILE *f;

    f = fopen("log.txt","a");
    if (f == NULL)
    {
        printf("Error opening log!\n");
        return(-1);
    }

    fprintf(f, "KnownPointsNum: %d QueryPointsNum: %d BlockNum: %d ThreadNumForBlock: %d Time: %f s\n", 
                KN, QN, nBlocks, nThreadsForBlock, gpuMeanTime);
    
    fclose(f);

    return 0;
}

int updateLogCpuGpu(float gpuMeanTime, float cpuMeanTime, float gpuSTD, float cpuSTD, int QN, int KN, int nBlocks, int nThreadsForBlock)
{
    FILE *f;

    f = fopen("fullLog.txt","a");
    if (f == NULL)
    {
        printf("Error opening log!\n");
        return(-1);
    }

    fprintf(f, "KnownPointsNum: %d QueryPointsNum: %d BlockNum: %d ThreadNumForBlock: %d CPUMeanTime: %f s CPUstd: %f GPUMeanTime: %f s GPUstd: %f\n", 
                KN, QN, nBlocks, nThreadsForBlock, cpuMeanTime, cpuSTD, gpuMeanTime, gpuSTD);
    
    fclose(f);

    return 0;
}

void getMaxAbsError(float *zValues, float *zValuesGPU, int QN, float *maxErr)
{
    int i;
    float err;

    *maxErr = -1;

    for (i = 0; i < QN; i++)
    {
        err = abs(zValues[i]-zValuesGPU[i]);
        if (err > 1e-6)
            printf("ERRORE\n%lf ; %lf\n\n",zValues[i],zValuesGPU[i]);
        if (err > *maxErr)
            *maxErr = err;
    }
}

float getRes(float *ref, float *result, int QN)
{
    int i;
    float res = 0, ref_norm = 0;

    for (i = 0; i < QN; i++)
    {
        ref_norm += ref[i]*ref[i];
    }

    ref_norm = sqrt(ref_norm);

    for (i = 0; i < QN; i++)
    {
        res += (ref[i]-result[i])*(ref[i]-result[i]);
    }

    return sqrt(res)/ref_norm;
}

float getSTD(float xm, float x[], int N)
{
    float s = 0;
    for (int i=0; i<N; i++)
    {
        s += pow(x[i] - xm,2);
    }

    s /= N-1;

    return sqrt(s);
}

