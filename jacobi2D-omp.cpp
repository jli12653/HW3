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
  double *uu = (double*) malloc((N+2)*(N+2) * sizeof(double)); // (N+2)^2 
	
  // creating the entire plane of points (N+2)*(N+2)
  #pragma omp parallel for
  for (long i = 0; i < (N+2)*(N+2); i++) {
  	if (i / (N+2) == 0 || i / (N+2) == N + 1 ){
		uu[i] = 0;	
	}
	else{
		if (i % (N+2) == 0 || i % (N+2) == N+1){ 
			uu[i] = 0;
		}
		else {
			uu[i] = u[(i/(N+2)-1)*N + (i % (N+2)-1)];
		}  
	}
  }

  #pragma omp parallel for
  for (long i = 0; i < N*N; i++) {
	long j = (i/N+1)*(N+2) + i % N + 1
	double U = 1.0/4*(h*h+ uu[j-1] + uu[j+1] + uu[j - (N+2)] + uu[j+(N+2)]);
  	u[i] = U;  
  }

  free(uu);
}


double residual(long N, double* u){
	double r = 0.0;
	
	for (long i=0; i<N*N; i++){
		double U = u[i]
		r += pow(U-1.0,2)	
  }	
	return sqrt(r);
} 

int main(int argc, char** argv) {
  const long N = 100000;

  //printf(" Iteration       Residual\n");
  
    
  double* u = (double*) malloc(N * sizeof(double)); // N
 

  // Initialize u
  for (long i = 0; i < N; i++) u[i] = 0.0;
  double Res = residual(N , u);
  double res = 0.0;
  double Rr = 0.0;
  int iter = 0;
  
  
  //printf("%10d %10f\n", iter, Res);

  // PART I
  //while (iter<5000 && Rr <= 1e4){
  //  	Gauss_Seidel(N, u);
  //  	res = residual(N,u);
  //  	Rr = Res/res;
  //  	iter++;
  //       printf("%10d %10f\n", iter, res);
  //}

  // PART II
  Timer t;
  t.tic();
  while (iter<100) {
    	Gauss_Seidel(N, u);
    	iter++;
  }
  double time = t.toc();
  printf("Time: %10f\n", time);

  free(u);

  return 0;
}


