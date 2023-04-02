#include <algorithm>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <string.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

// Given that we are using f = 1, so I ingore the term f, just replace it with 1.
void Jacobi(int N, double *u) {
  double h = 1.0 / ( N + 1.0 );
  double hsq = h*h;
  double *uu = (double*) malloc( (N+2)*(N+2) * sizeof(double)); // (N+2)^2 
  int k, up, down, left, right;
	
//   #pragma omp parallel for
//   for (int i = 0; i < N+2; i++) {
//   	uu[i] = uu[(N+2)*(N+1)+i] = 0.0;
// 	uu[i*(N+2)] = uu[i*(N+2) + N+1] =0.0;
//   }
  	
  #pragma omp parallel for
  for (int i = 1; i <=N; i++) {
	for (int j = 1; j <=N; j++) {
		k = i * (N + 2) + j;
  		up = k + N + 2;
		down = k - N - 2;
		left = k - 1;
		right = k + 1;
	
	
		double U_up = u[up];
		
	
		double U_left = u[left];
	
	
		double U_right = u[right];
	
	
		double U_down = u[down];
		
        	uu[k] = 1.0/4*(hsq + U_up + U_down + U_right + U_left);
	}
  }
	
   printf("hsq is %d  \n", hsq);
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
  for (int i = 1; i <=N; i++) {
	for (int j = 1; j <=N; j++) {
		k = i * (N + 2) + j;
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
  int k, up, down, left, right;
  double U_up, U_down, U_left, U_right, U;

  #pragma omp parallel for reduction (+:r)
  for (int i = 1; i <=N; i++) {
	for (int j = 1; j <=N; j++) {
		k = i * (N + 2) + j;
		up = k + N + 2;
		down = k - N -2;
		left = k - 1;
		right = k + 1;
	  
		U = u[k];
	
	
	 	U_up = u[up];
		
	
		U_left = u[left];
	
	
		U_right = u[right];
	

		U_down = u[down];
	
	
		temp = (4.0*U - U_up - U_down - U_left - U_right)*invhsq - 1.0;
	  
  		r += temp * temp; 
	}
  }

	
  return sqrt(r);
}

int main(int argc, char** argv) {
  int k = 0;
  int N = 4;

  printf(" Iteration       Residual\n");
  
    
  double* u = (double*) malloc((N+2) * (N+2) * sizeof(double)); // N
 

  // Initialize u
  for (int i = 0; i < (N+2)*(N+2); i++){ double z = 0.0; u[i] = z;}

//   for (int i = 0; i < (N+2)*(N+2); i++) printf("%d  ", u[k]);
//   printf("\n");
  printf("u[0] is %d  \n", u[0]);
  printf("=============================================================\n");
  for (int i = 0; i <=N+1; i++) {
	for (int j = 0; j <=N+1; j++) {
		k = i * (N + 2) + j;
		printf("%d  ", u[k]);
	}
	printf("\n");
  }
  printf("=============================================================\n");
  printf("u[0] is %d  \n", u[0]);
	
  double Res = residual(N , u);
  double res = 0.0;
  double Rr = 0.0;
  int iter = 0;
  
  
  printf("%10d %10f\n", iter, Res);

  while (iter<10 && Rr <= 1e4){
   	Jacobi(N, u);
	  
	printf("=============================================================\n");
  	for (int i = 0; i <=N+1; i++) {
		for (int j = 0; j <=N+1; j++) {
			k = i * (N + 2) + j;
			printf("%d  ", u[k]);
	}
		printf("\n");
  	}
  	printf("=============================================================\n");  
	  
	
   	res = residual(N,u);
   	Rr = Res/res;
   	iter++;
        printf("%10d %10f\n", iter, res);
  }

  double tt = omp_get_wtime();
  iter = 0;
  while (iter<100) {
    	Jacobi(N, u);
    	iter++;
  }
  double time = omp_get_wtime() - tt;
  printf("Time: %10f\n", time);

  free(u);

  return 0;
}


