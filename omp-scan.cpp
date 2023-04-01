#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  //printf("maximum number of threads = %d\n", omp_get_max_threads());
  long* correction = (long*) malloc(omp_get_max_threads() * sizeof(long));
  
  if (n == 0) return;
  prefix_sum[0] = 0;
  
  #pragma omp parallel
  {
    int p = omp_get_num_threads();
    int t = omp_get_thread_num();
    //printf("hello world from thread %d of %d\n", p, t);
    
    long s = 0;
    #pragma omp for schedule(static)
    for (long i = 0; i < n-1; i++) {
      s += A[i];
      prefix_sum[i+1] = s;
    }
    correction[t] = s;
    //printf("correction from thread %d of %d is %d\n", p, t, correction[t]);
    
    long offset = 0;
    
    for (int i = 0; i < t; i++){
      offset += correction[i];
    }
    printf("offset from thread %d of %d is %d\n", p, t, offset);
    
    #pragma omp for schedule(static)
    for (long i = 1; i < n; i++) {
      prefix_sum[i] += offset;
    }
  }
  
  free(correction);
  // Fill out parallel scan: One way to do this is array into p chunks
  // Do a scan in parallel on each chunk, then share/compute the offset
  // through a shared vector and update each chunk by adding the offset
  // in parallel
}

int main() {
  long N = 10;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();
  for (long i = 0; i < N; i++) B1[i] = 0;
  
  for (long i = 0; i < N; i++) printf(A[i]);;
  
  
  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
