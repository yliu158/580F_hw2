#include <immintrin.h>
#include <cmath>
#include <functional>
#include <chrono>
#include <random>
#include <iostream>
#include <cassert>
#include <omp.h>
using namespace std;

#define NUM_DIMENSIONS 16
const int N = 100;
const int num_hist_intervals = 10;
alignas(32) static float points[N][NUM_DIMENSIONS];

int main() {

  uniform_real_distribution<float> dist(-1, 1);
  static int count = 0;
  static int rejected = 0;
  static int threads_num = omp_get_num_threads();
  double start = omp_get_wtime();

  #pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    default_random_engine eng(thread_id);
    float distance_sq_sum;
    float temp[NUM_DIMENSIONS];

  #pragma omp for
    for (int i = thread_id; i < N; i+=threads_num ) {
      distance_sq_sum = 2.0;
      while (distance_sq_sum > 1.0) {
        distance_sq_sum = 0.0;
        for (int j = 0; j < NUM_DIMENSIONS; ++j) {
          temp[j] = dist(eng);
          distance_sq_sum += temp[j]*temp[j];
        }
        ++ rejected;
      }
      for (int k = 0; k < NUM_DIMENSIONS; ++k) {
        points[i][k] = temp[k];
      }
      ++ count;
      printf("thread_id: %d  count: %d rejected: %d\n",thread_id ,count, rejected-count);
    }

  }
  printf("rejected ratio: %f\n", (float)(rejected-count)/(float)(rejected));
  double end = omp_get_wtime();
  printf("sampling time: %f\n", end-start);



  return 0;
}
