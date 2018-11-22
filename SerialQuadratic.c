#include <stdio.h>
#include <math.h>

int main()
{
   int i;
   float steps = 200, first, second, third;
   int dataPoints = 1000;
   float X[dataPoints], Y[dataPoints];
   float discretePoint = steps/dataPoints;
   //discretise the range to work out X[i]
   for (i= 0; i < dataPoints+1; i++){
      X[i] = (discretePoint * i)-100;
   }

   //Work out the value for Y[i]
   for(i=0; i < dataPoints+1; i++)
   {
      first = ((X[i]-2))*((X[i]-2));
      second = (pow((X[i]-6),2)/10);
      third = (1/(pow(X[i],2)+1));
      Y[i] = (exp(-first)+exp(-second)+third);
   }

   //print out the values at the moment
   for(i=0; i < dataPoints+1; i++)
   {
      printf("X = %0.8f \n", X[i]);
      printf("Y = %0.8f \n", Y[i]);
   }
   printf("%f \n", discretePoint);
}