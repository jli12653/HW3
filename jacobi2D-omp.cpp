#include <stdio.h>
#include "utils.h"
#include <cmath>


void Jacobi( long N, double *u) {
  double h = 1.0/N;
  double *uu = (double*) malloc(N * sizeof(double)); // N 
  
  for (int i = 0; i < N; i++) {
	
	if (i == 0){ 
	double U = u[i+1];
	double UU = 1.0/2 * (h*h +  U);
	uu[i] = UU;
	}
	else if (i == N-1){ 
	double U = u[i-1];
	double UU = 1.0/2 * (h*h +  U);
	uu[i] = UU;
	}
  	else {
	double U = u[i-1];
	double UU = u[i+1];
	double UUU = 1.0/2 * (h*h + U +  UU);
	uu[i] = UUU;
	}  
  }

  for (int i = 0; i < N; i++) {
	double U = uu[i];
  	u[i] = U;  
  }

  free(uu);
}

void  Gauss_Seidel( long N, double *u) {
  double h = 1.0/N;
  double *uu = (double*) malloc(N * sizeof(double)); // N 


  for (int i = 0; i < N; i++) {
  	if (i == 0) {double U = u[i+1]; double UU = 1.0/2 * (h*h + U); uu[i]= UU;}
	else if (i == N-1){double U = uu[i-1]; double UU = 1.0/2 * (h*h + U); uu[i] = UU; }
	else{double U = uu[i-1]; double UU = u[i+1]; double UUU = 1.0/2 * (h*h + U + UU); uu[i] = UUU;}  
  }

  for (int i = 0; i < N; i++) {
	double U = uu[i];
  	u[i] = U;  
  }


  free(uu);
}

double residual(long N, double* u){
	double h = 1.0/N;
	double r = 0.0;
	
	
	for (int i=0; i<N; i++){
		if (i == 0) {double U=u[i]; double UU = u[i+1]; r += pow(((2*U-UU)/h/h-1.0),2);}
		else if (i == N-1) {double U=u[i]; double UU = u[i-1]; r += pow(((2*U-UU)/h/h-1.0),2);}
		else {double U=u[i]; double UU = u[i+1]; double UUU = u[i-1]; r += pow(((-UUU+2*U-UU)/h/h-1.0),2);}
          	
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


