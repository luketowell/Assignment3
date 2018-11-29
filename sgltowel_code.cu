#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

// Code for the parallel CUDA code
__global__ void exponentialFunction (int dataPoints, float *devX, float *devY)
{
   float first, second, third;
   int my_i = blockIdx.x*blockDim.x + threadIdx.x;
	
   if (my_i<dataPoints+1){
      first = ((devX[my_i]-2))*((devX[my_i]-2));
      second = (pow((devX[my_i]-6.0),2)/10);
      third = (1/(pow((double)devX[my_i],2.0)+1)); 
      devY[my_i] = (expf(-first)+expf(-second)+third);
   }
	
}

// Serial Code
void serialFunction (int dataPoints, float *serialX, float *serialY)
{
   float first, second, third;
   int i;

   for(i=0; i < dataPoints+1; i++)
      {
         first = ((serialX[i]-2))*((serialX[i]-2));
         second = (pow((serialX[i]-6),2)/10);
         third = (1/(pow(serialX[i],2)+1));
         serialY[i] = (exp(-first)+exp(-second)+third);
      }

   printf("ran serial code \n");
}

int main(int argc, char **argv)
{
   int i, numGPU;
	
   cudaGetDeviceCount(&numGPU);
   if (numGPU >= 1) {

      float steps = 200;
      int dataPoints = strtol(argv[1], NULL, 10);
     
      // X and F(x) as Y declaration
      float *X, *Y, *serialX, *serialY;
      float *devX, *devY;
      float maxY, serialMaxY;
  
      //create cuda timing objects
      cudaEvent_t startCuda, stopCuda;
      cudaEventCreate(&startCuda);
      cudaEventCreate(&stopCuda);

      //OMP timing variables
      double cudaStart, cudaEnd, serialFunctionStart, serialFunctionEnd, serialStart, serialEnd, serialInitStart, serialInitEnd, ompInitStart, ompInitEnd;
      
      // Device memory allocation
      cudaMalloc(&devX, dataPoints*sizeof(float));
      cudaMalloc(&devY, dataPoints*sizeof(float));

      //Host Memory Allocation
      X = (float *) malloc(sizeof(float)*dataPoints);
      Y = (float *) malloc(sizeof(float)*dataPoints);
      serialX = (float *) malloc(sizeof(float)*dataPoints);
      serialY = (float *) malloc(sizeof(float)*dataPoints);
      
      //Start executing
      cudaStart = omp_get_wtime();	
      float discretePoint = steps/dataPoints;


      //discretise the range to work out X[i]
      ompInitStart = omp_get_wtime();
      #pragma omp parallel for default(none) shared(dataPoints, X, discretePoint) private(i)
      for (i= 0; i < dataPoints+1; i++){
         X[i] = (discretePoint * i)-100;
      }
      ompInitEnd = omp_get_wtime();

      // Copy the host contents of X over to device devX
      cudaMemcpy(devX, X, dataPoints*sizeof(float), cudaMemcpyHostToDevice);	

      // Check for errors after Copying X over to new Device
      cudaError err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("(1) CUDA RT error: %s \n", cudaGetErrorString(err));
      }
  
      //Work out threads and blocks and print out number of threads and blocks
      int threads = strtol(argv[2], NULL, 10);
      int blocks = ceil((float)dataPoints/(float)threads);
      printf("using %d threads on %d blocks \n", threads, blocks);


      //Start the Cuda Timings
      cudaEventRecord(startCuda, 0);

      //Call the function kernel
      exponentialFunction<<<blocks,threads>>> (dataPoints, devX, devY);
      //Stop the Cuda Timings
      cudaEventRecord(stopCuda, 0);
   
      // check for errors after running Kernel
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("(2) CUDA RT error: %s \n", cudaGetErrorString(err));
      }

      // Copy over the Y value from the device to the host
      cudaMemcpy(Y, devY, dataPoints*sizeof(float), cudaMemcpyDeviceToHost);
   
      //Check for errors after copying errors over from device to host.
      err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("(3) CUDA RT error: %s \n", cudaGetErrorString(err));
      }

      //clean up memory
      cudaFree(devX);
      cudaFree(devY);
	
      cudaEnd = omp_get_wtime();
      //Work out time
      float cTime;
      cudaEventElapsedTime(&cTime, startCuda, stopCuda);
      
 
      //print out the Cuda+OMP result and timing
      for(i=0; i < dataPoints+1; i++)
      {
         printf("X = %0.5f \n", X[i]);
         printf("Y = %0.5f \n", Y[i]);
      }

      //start serial timings
      serialStart = omp_get_wtime();
      
      //work out discrete point again for serial
      discretePoint = steps/dataPoints;


      //discretise the range to work out X[i]
      serialInitStart = omp_get_wtime();

      for (i= 0; i < dataPoints+1; i++){
         X[i] = (discretePoint * i)-100;
      }
      
      serialInitEnd = omp_get_wtime();

      //call the serial code:
      serialFunctionStart = omp_get_wtime(); 
      serialFunction(dataPoints, X, Y);
      serialFunctionEnd = omp_get_wtime();

      //work out max in serial

      //end serial timings
      serialEnd = omp_get_wtime();
      
      //total timings
      printf("omp init %0.5f\n", (ompInitEnd - ompInitStart)*1000);
      printf("cuda function: %0.5f\n", cTime);
      printf("total cuda Time: %0.5f\n", (cudaEnd - cudaStart)*1000);
      printf("serial init %0.5f\n", (serialInitEnd - serialInitStart)*1000);
      printf("serial function: %0.5f\n", (serialFunctionEnd-serialFunctionStart)*1000);
      printf("all serial: %0.5f \n", (serialEnd - serialStart) * 1000);

   }
   else
   {
    printf("No GPUs are detected!\n");
   }
}