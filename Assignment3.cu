#include <stdio.h>
#include <math.h>

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
void serialFunction (int dataPoints, float *X, float *Y)
{
   float first, second, third;
   int i;

   for(i=0; i < dataPoints+1; i++)
      {
         first = ((X[i]-2))*((X[i]-2));
         second = (pow((X[i]-6),2)/10);
         third = (1/(pow(X[i],2)+1));
         Y[i] = (exp(-first)+exp(-second)+third);
      }

   printf("ran serial code \n");
}

int main()
{
   int i, numGPU;
	
   cudaGetDeviceCount(&numGPU);
   if (numGPU >= 1) {

      float steps = 200;
      int dataPoints = 1000;

      // X and F(x) as Y declaration
      float *X, *Y;
      float *devX, *devY;
      //Insert here if statement for if not detected;
  
      //create cuda timing objects
      cudaEvent_t startCuda, stopCuda;
      cudaEventCreate(&startCuda);
      cudaEventCreate(&stopCuda);
      // Device memory allocation
      cudaMalloc(&devX, dataPoints*sizeof(float));
      cudaMalloc(&devY, dataPoints*sizeof(float));

      //Host Memory Allocation
      X = (float *) malloc(sizeof(float)*dataPoints);
      Y = (float *) malloc(sizeof(float)*dataPoints);
	
      float discretePoint = steps/dataPoints;


      //discretise the range to work out X[i]
      for (i= 0; i < dataPoints+1; i++){
         X[i] = (discretePoint * i)-100;
      }

      // Copy the host contents of X over to device devX
      cudaMemcpy(devX, X, dataPoints*sizeof(float), cudaMemcpyHostToDevice);	

      // Check for errors after Copying X over to new Device
      cudaError err = cudaGetLastError();
      if (err != cudaSuccess) {
        printf("(1) CUDA RT error: %s \n", cudaGetErrorString(err));
      }
  
      //Work out threads and blocks and print out number of threads and blocks
      int threads = 8;
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
  
      //Work out time
      float cTime;
      cudaEventElapsedTime(&cTime, startCuda, stopCuda);
      printf("Ran on device in: %f microseconds \n", cTime);
 
      //print out the Cuda+OMP result and timing
      /*for(i=0; i < dataPoints+1; i++)
      {
         printf("X = %0.5f \n", X[i]);
         printf("Y = %0.5f \n", Y[i]);
      }*/

      //call the serial code:
      serialFunction(dataPoints, X, Y);
      printf("%f \n", discretePoint);
   }
   else
   {
    //No GPU detected  
    printf("No GPUs are detected!\n");
   }
}