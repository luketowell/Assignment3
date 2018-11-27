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

//run Serial Code
void serialFunction (int dataPoints, float *X, float *Y)
{  
   float first, second, third;
   int i;
   printf("running serial code \n");
   for(i=0; i < dataPoints+1; i++)
      {  
         first = ((X[i]-2))*((X[i]-2));
         second = (pow((X[i]-6),2)/10);
         third = (1/(pow(X[i],2)+1));
         Y[i] = (exp(-first)+exp(-second)+third);
      }
   
   for(i=0; i < dataPoints+1; i++)
   {  
      printf("X = %0.8f \n", X[i]);
      printf("Y = %0.8f \n", Y[i]);
   }
   
   printf("ran serial code \n");
}

int main()
{
   int i;
   float steps = 200;
   int dataPoints = 1000;

   // X and F(x) as Y declaration
   float *X, *Y;
   float *devX, *devY;
   //Insert here if statement for if not detected;

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

   //Work out threads and blocks
   int threads = 32;
   int blocks = ceil((float)dataPoints/(float)threads);

   //Call the function kernel
   exponentialFunction<<<blocks,threads>>> (dataPoints, devX, devY);

   //wait for the device
   cudaThreadSynchronize();

   // Copy over the Y value from the device to the host
   cudaMemcpy(Y, devY, dataPoints*sizeof(float), cudaMemcpyDeviceToHost);

   //Check for Errors
   cudaError err = cudaGetLastError();
   if (err != cudaSuccess) {
     printf("(1) CUDA RT error: %s \n", cudaGetErrorString(err));
   }

   //clean up memory
   cudaFree(devX);
   cudaFree(devY);

   //print out the values at the moment
   for(i=0; i < dataPoints+1; i++)
   {
      printf("X = %0.8f \n", X[i]);
      printf("Y = %0.8f \n", Y[i]);
   }

   //call the serial code:
   serialFunction(dataPoints, X, Y);
   printf("%f \n", discretePoint);
}