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
	
// #pragma omp parallel
// {	
#pragma omp parallel for
  for (int i = 1; i <=N; i++) {

	  for (int j = 1; j <=N; j++) {
      int k = i*(N+2) + j;
  	
      // double U_up = u[k + N + 2];
      // double U_left = u[k - 1];
      // double U_right = u[k + 1];
      // double U_down = u[k - N - 2];

    //+ u[up] + u[left] + u[right] + u[down]
    //+ U_up + U_left + U_right + U_down
    //u[i + N + 2] + u[i - 1] + u[i + 1] + u[i - N - 2]
		
      uu[k] = 0.25*(hsq + u[k + N + 2] + u[k - 1] + u[k + 1] + u[k - N - 2]);

	}
}

	

#pragma omp parallel for
  for (int i = 1; i <=N; i++) {
	  for (int j = 1; j <=N; j++) {
      int k = i*(N+2) + j;
      double U = uu[k];
      u[k] = U;
	}  
  }
  

  free(uu);
}


double residual(int N, double* u){
  double h = 1.0/(N+1);
  double invhsq = 1.0/h/h;
  double r, temp = 0.0;
	
#pragma omp parallel for reduction (+:r)
  for (int i = 1; i <=N; i++) {
//#pragma omp parallel for 
    for (int j = 1; j <=N; j++) {
      int k = i * (N + 2) + j;
    
      temp = ((4.0*u[k] - u[k + N + 2] - u[k - N -2] - u[k - 1] - u[k + 1])*invhsq - 1.0);
      
      r += temp * temp; 
	}
  }

	
  return sqrt(r);
}

int main(int argc, char** argv) {
  int k = 0;
  int N = 10000;

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

  // while (iter<5000 && Res/res < 1e4){
  //  	Jacobi(N, u);
	  
	
  //  	res = residual(N,u);
  //  	iter++;
  //   printf("%10d %10f\n", iter, res);
  // }

	
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


