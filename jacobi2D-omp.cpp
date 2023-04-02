#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <cmath>

#if defined(_OPENMP)
#include <omp.h>
#endif

// Given that we are using f = 1, so I ingore the term f, just replace it with 1.
void Jacobi(long N, double *u) {
  double h = 1.0/(N+1);
  double *uu = (double*) malloc(N*N * sizeof(double)); // (N+2)^2 
  long up, down, left, right;
	
  // creating the entire plane of points (N+2)*(N+2)
  #pragma omp parallel for
  for (long i = 0; i < N*N; i++) {
  	up = i + N;
	down = i - N;
	left = i % N - 1;
	right = i % N + 1;
	
	double U_up, U_down, U_left, U_right = 0.0;
	
	if (up >= N^2) U_up = 0;
	else U_up = u[up];
		
	if (left < 0) U_left = 0;
	else U_left = u[i-1];
	
	if (right >= N) U_right = 0;
	else U_right = u[i+1];
	
	if (down < 0) U_down = 0;
	else U_down = u[down];
		
        uu[i] = 1.0/4*(h*h + U_up + U_down + U_right + U_left);
  }

  #pragma omp parallel for
  for (long i = 0; i < N*N; i++) {
	double NewU = uu[i];
  	u[i] = NewU;  
  }

  free(uu);
}


double residual(long N, double* u){
  double h = 1.0/(N+1);
  double r, temp = 0.0;
  long up, down, left, right;
  double U_up, U_down, U_left, U_right, U = 0.0;

  #pragma omp parallel for reduction (+:r)
  for (long i = 0; i < N*N; i++) {
	up = i + N;
	down = i - N;
	left = i % N - 1;
	right = i % N + 1;
	  
	U = u[i];
	
	if (up >= N^2) U_up = 0;
	else U_up = u[up];
		
	if (left < 0) U_left = 0;
	else U_left = u[i-1];
	
	if (right >= N) U_right = 0;
	else U_right = u[i+1];
	
	if (down < 0) U_down = 0;
	else U_down = u[down];
	
	
	temp = (4.0*U - U_up - U_down - U_left - U_right)/h/h - 1.0;
	printf("%d",temp);
	  
  	r += temp * temp;  
  }

	
  return sqrt(r);
}

int main(int argc, char** argv) {
  const long N = 10;

  printf(" Iteration       Residual\n");
  
    
  double* u = (double*) malloc(N * sizeof(double)); // N
 

  // Initialize u
  for (long i = 0; i < N; i++) u[i] = 0.0;
  double Res = residual(N , u);
  double res = 0.0;
  double Rr = 0.0;
  int iter = 0;
  
  
  printf("%10d %10f\n", iter, Res);

  while (iter<5000 && Rr <= 1e4){
   	Jacobi(N, u);
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


