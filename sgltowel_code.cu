#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

// Code for the parallel CUDA code
__global__ void exponentialFunction (int dataPoints, float *X, float *Fx)
{
   float first, second, third;
   int my_i = blockIdx.x*blockDim.x + threadIdx.x;
	
   if (my_i<dataPoints+1){
      first = ((X[my_i]-2))*((X[my_i]-2));
      second = (pow((X[my_i]-6.0),2)/10);
      third = (1/(pow((double)X[my_i],2.0)+1)); 
      Fx[my_i] = (expf(-first)+expf(-second)+third);
   }
	
}

__global__ void initX (float *X, int dataPoints, float discretePoint, int threads)
{
    int my_i = blockIdx.x*blockDim.x + threadIdx.x;
 
    if (my_i < dataPoints+1){
        X[my_i] = (discretePoint * my_i)-100; 
    }
}

int main(int argc, char **argv)
{   
   int i, numGPU;
   // Steps between the two outer values e.g. -100 and +100.
   float steps = 200;
   // set the amount of dataPoints that you want to discretize the function over
   int dataPoints = strtol(argv[1], NULL, 10);

   //all serial code
   float *serialX, *serialFx;
   float serialMaxFx, serialFirst, serialSecond, serialThird;

   double serialFunctionStart, serialFunctionEnd, serialStart, serialEnd, serialInitStart, serialInitEnd, serialMaxStart, serialMaxEnd;
   
   serialX = (float *) malloc(sizeof(float)*dataPoints);
   serialFx = (float *) malloc(sizeof(float)*dataPoints);
   //start serial timings
   serialStart = omp_get_wtime();
   
   //work out discrete point again for serial
   float serialDiscretePoint = steps/dataPoints;

   //discretise the range to work out X[i]
   serialInitStart = omp_get_wtime();

   for (i= 0; i < dataPoints+1; i++){
     serialX[i] = (serialDiscretePoint * i)-100;
   }
   
   serialInitEnd = omp_get_wtime();

   //work out F(x) as Fx serial code:
   serialFunctionStart = omp_get_wtime(); 
   
   for(i=0; i < dataPoints+1; i++)
   {
      serialFirst = ((serialX[i]-2))*((serialX[i]-2));
      serialSecond = (pow((serialX[i]-6),2)/10);
      serialThird = (1/(pow(serialX[i],2)+1));
      serialFx[i] = (exp(-serialFirst)+exp(-serialSecond)+serialThird);
   }
   serialFunctionEnd = omp_get_wtime();
   
   serialMaxStart = omp_get_wtime();
   //work out max in serial
   for(i=0; i < dataPoints+1; i++)
   {
      if (serialFx[i] > serialMaxFx){
          serialMaxFx = serialFx[i];
      }
    }
   serialMaxEnd = omp_get_wtime();

   //end serial timings
   serialEnd = omp_get_wtime();
   
   //Start cuda code execution
   cudaGetDeviceCount(&numGPU);
   if (numGPU >= 1) {
      //set number of omp threads to be used;
      int numOMPThreads = strtol(argv[3], NULL, 10); 
      omp_set_num_threads(numOMPThreads);
 
      // X and F(x) as Fx declaration
      float *X, *Fx;
      float *devX, *devFx;
      float maxFx;
  
      //create cuda timing objects
      cudaEvent_t cudaIStart, cudaIEnd, startCuda, stopCuda;
      cudaEventCreate(&cudaIStart);
      cudaEventCreate(&cudaIEnd);
      cudaEventCreate(&startCuda);
      cudaEventCreate(&stopCuda);
      
      //declare and work out the number of threads and blocks.
      int potentialBlocks, maxBlocks = 65535; 
      int threads = strtol(argv[2], NULL, 10);

      // Ternary statement for working out amout of blocks to be used.
      potentialBlocks = ceil((float)dataPoints/(float)threads);
      int blocks = (potentialBlocks<maxBlocks) ? potentialBlocks : maxBlocks ;
      
      //statement to ensure enough threads are used.
      
      if(blocks==maxBlocks){
        int minThreads = ceil((float)dataPoints/(float)blocks);
        threads = minThreads;
        printf("You have not requested enough threads for the program to execute accurately.\n We are using the minimum number of threads required which for your data size is : %d\n", minThreads);
      }	


      printf("Discretizing the function across %d data points using %d threads on %d blocks \n", dataPoints, threads, blocks);

      //OMP timing variables
      double cudaStart, cudaInitEnd,cudaFuncMemStart, cudaFuncMemEnd, cudaEnd;
      double ompMaxStart, ompMaxEnd;
      
      // Device memory allocation
      cudaMalloc(&devX, dataPoints*sizeof(float));
      cudaMalloc(&devFx, dataPoints*sizeof(float));

      //Host Memory Allocation
      X = (float *) malloc(sizeof(float)*dataPoints);
      Fx = (float *) malloc(sizeof(float)*dataPoints);
      
      
      //Start executing
      cudaStart = omp_get_wtime();	
      float discretePoint = steps/dataPoints;

      //init the values of X cuda timings
      cudaEventRecord(cudaIStart,0);
      initX<<<blocks, threads>>>(devX, dataPoints, discretePoint, threads);
      cudaEventRecord(cudaIEnd);
      cudaEventSynchronize(cudaIEnd);
      cudaMemcpy(X, devX, dataPoints*sizeof(float), cudaMemcpyDeviceToHost);
      cudaInitEnd = omp_get_wtime();	

      cudaFuncMemStart = omp_get_wtime();
      // Copy the host contents of X over to device devX
      cudaMemcpy(devX, X, dataPoints*sizeof(float), cudaMemcpyHostToDevice);	
    
      // Check for errors after Copying X over to new Device
      cudaError err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("(1) CUDA RT error: %s \n", cudaGetErrorString(err));
      }

      //Start the Cuda Timings
      cudaEventRecord(startCuda, 0);

      //Call the function kernel
      exponentialFunction<<<blocks,threads>>> (dataPoints, devX, devFx);
      //Stop the Cuda Timings
      cudaEventRecord(stopCuda);
      cudaEventSynchronize(stopCuda);
      // check for errors after running Kernel
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("(2) CUDA RT error: %s \n", cudaGetErrorString(err));
      }

      // Copy over the Fx value from the device to the host
      cudaMemcpy(Fx, devFx, dataPoints*sizeof(float), cudaMemcpyDeviceToHost);
      cudaFuncMemEnd=omp_get_wtime();
      //Check for errors after copying errors over from device to host.
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("(3) CUDA RT error: %s \n", cudaGetErrorString(err));
      }

      //clean up device memory
      cudaFree(devX);
      cudaFree(devFx);
	
      //Work out time
      //CUDA initialisation timing
      float iTime;
      cudaEventElapsedTime(&iTime, cudaIStart, cudaIEnd);
      //CUDA function timing
      float cTime;
      cudaEventElapsedTime(&cTime, startCuda, stopCuda);
      
      ompMaxStart = omp_get_wtime();
      //print out the Cuda+OMP result
      #pragma omp parallel for default(none) shared(Fx, dataPoints) private(i) reduction(max: maxFx) 
      for(i=0; i < dataPoints+1; i++)
      {
         if (Fx[i] > maxFx){
             maxFx = Fx[i];
         }
      }
      ompMaxEnd = omp_get_wtime();
      //end of cuda+omp implementation
      cudaEnd = omp_get_wtime();

      //total timings
      printf("~~~~~~~~~~~~~~~~~~~~~~~~~ Results of Run ~~~~~~~~~~~~~~~~~~~~~~~\n");
      printf("|  Program Step  | Kernel + Mem  |    Kernel   |    Serial     |\n");
      printf("|______________________________________________________________|\n");
      printf("| Init of X      |  %0.8f  |  %0.8f |  %0.8f |\n", (cudaInitEnd - cudaStart)*1000, iTime, (serialInitEnd - serialInitStart)*1000);
      printf("| Calculate F(x) |  %0.8f  | %0.8f | %0.8f |\n", (cudaFuncMemEnd - cudaFuncMemStart)*1000, cTime, (serialFunctionEnd-serialFunctionStart)*1000);
      printf("| Max of F(x)    | ~~~~ N/A ~~~~ | %0.8f |   %0.8f |\n", (ompMaxEnd - ompMaxStart)*1000, (serialMaxEnd - serialMaxStart)*1000);
      printf("|______________________________________________________________|\n");
      printf("|                |      CUDA + OMP      |      Serial Code     |\n", (ompMaxEnd - ompMaxStart)*1000, (serialMaxEnd - serialMaxStart)*1000);
      printf("| Total Time     |     %0.8f     |     %0.8f    |\n", (cudaEnd - cudaStart)*1000, (serialEnd - serialStart) * 1000);
      printf("| Max Value F(x) |       %0.8f     |        %0.8f    |\n", maxFx, serialMaxFx); 
      printf("|______________________________________________________________|\n");
   }
   else
   {
    printf("serial init %0.5f\n", (serialInitEnd - serialInitStart)*1000);
    printf("Unfortunately no GPUs are detected!\n");
    printf("Here are the serial results!\n");
    printf("serial function: %0.5f\n", (serialFunctionEnd-serialFunctionStart)*1000);
    printf("serial max calc: %0.5f\n", (serialMaxEnd - serialMaxStart)*1000);
    printf("all serial: %0.5f \n", (serialEnd - serialStart) * 1000);
    printf("serial maxFx: %0.8f\n", serialMaxFx);
   }
}