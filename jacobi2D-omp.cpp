// g++ -fopenmp jacobi2D-omp.cpp && ./a.out
// g++ -std=c++11 -O3 jacobi2D-omp.cpp && ./a.out



#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#else
#include "utils.h"
#endif

// Given that we are using f = 1, so I ingore the term f, just replace it with 1.
void Jacobi(int N, double *u) {
  double h = 1.0 / ( N + 1 );
  double hsq = h*h;
  double *uu = (double*) malloc( (N+2)*(N+2) * sizeof(double)); // (N+2)^2 
  int  up, down, left, right;
	
#pragma omp parallel
{	
#pragma omp parallel for schedule(static)
  for (int i = N+3; i <= N*N+3*N; i++){
  		up = i + N + 2;
		  down = i - N - 2;
		  left = i - 1;
		  right = i + 1;
	
	
		double U_up = u[up];
		
	
		double U_left = u[left];
	
	
		double U_right = u[right];
	
	
		double U_down = u[down];
		
    uu[i] = 0.25*(hsq + u[up] + u[down] + u[right] + u[left]);
	}
}

	
//   printf("=============================================================\n"); 
//   for (int i = 0; i <=N+1; i++) {
// 	for (int j = 0; j <=N+1; j++) {
// 		k = i * (N + 2) + j;
// 		printf("%d  ", uu[k]);
// 	}
// 	printf("\n");
//   }
//   printf("=============================================================\n");
//   printf("u[0] is %d  \n", u[0]);

#pragma omp parallel for
  for (int i = N+3; i <= N*N+3*N; i++){
  // for (int i = 1; i <=N; i++) {
  //   for (int j = 1; j <=N; j++) {
      //k = i * (N + 2) + j;
      double U = uu[i];
      u[i] = U;
	}
  

  free(uu);
}


double residual(int N, double* u){
  double h = 1.0/(N+1);
  double invhsq = 1.0/h/h;
  double r, temp = 0.0;
  int k, up, down, left, right;
	
//#pragma omp parallel for reduction (+:r)
  for (int i = 1; i <=N; i++) {
	for (int j = 1; j <=N; j++) {
		k = i * (N + 2) + j;
		up = k + N + 2;
		down = k - N -2;
		left = k - 1;
		right = k + 1;
	
		temp = ((4.0*u[k] - u[up] - u[down] - u[left] - u[right])*invhsq - 1.0);
	  
  		r += temp * temp; 
	}
  }

	
  return sqrt(r);
}

int main(int argc, char** argv) {
  int k = 0;
  int N = 10;

  printf(" Iteration       Residual\n");
  
    
  double* u = (double*) malloc((N+2) * (N+2) * sizeof(double)); // N
 

  // Initialize u
  for (int i = 0; i < (N+2)*(N+2); i++) u[i] = 0.0;
  
//   for (int i = 1; i <=N; i++) {
// 	for (int j = 1; j <=N; j++) {
// 		k = i * (N + 2) + j;
// 		u[k] = 0.5;
// 	}
//   }

//   printf("=============================================================\n");
//   for (int i = 0; i <=N+1; i++) {
// 	for (int j = 0; j <=N+1; j++) {
// 		k = i * (N + 2) + j;
// 		printf("%f  ", u[k]);
// 	}
// 	printf("\n");
//   }
//   printf("=============================================================\n");
	
  double Res = residual(N , u);
  double res = Res;
  int iter = 0;
  
  
  printf("%10d %10f\n", iter, Res);

  while (iter<5000 && Res/res < 1e4){
   	Jacobi(N, u);
	  
	
   	res = residual(N,u);
   	iter++;
    printf("%10d %10f\n", iter, res);
  }

  	printf("=============================================================\n");
  	for (int i = 0; i <=N+1; i++) {
		for (int j = 0; j <=N+1; j++) {
			k = i * (N + 2) + j;
			printf("%f  ", u[k]);
	}
		printf("\n");
  	}
  	printf("=============================================================\n");  	
	
#ifdef _OPENMP
  double t = omp_get_wtime();
#else
  Timer tt;
  tt.tic();
#endif

  iter = 0;
  while (iter<100) {
    	Jacobi(N, u);
    	iter++;
  }
#ifdef _OPENMP
  t = omp_get_wtime() - t;
#else
  double t = tt.toc();
#endif
  printf("Time: %10f\n", t);

  free(u);

  return 0;
}


