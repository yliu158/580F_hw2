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
const int N = 100'000;
alignas(32) static float points[N][NUM_DIMENSIONS];
// alignas(32) static float temp[NUM_DIMENSIONS];

int main() {

  uniform_real_distribution<float> dist(-1, 1);

  // static float distance_sq_sum;
  static int count = 0;
  static int notincluded = 0;

  double start = omp_get_wtime();
  // omp_set_num_threads(4);
  #pragma omp parallel
  {
    default_random_engine eng;
    while (count < N) {
      float distance_sq_sum = 0.0;
      #pragma omp for
      for (int i = 0; i < NUM_DIMENSIONS; ++i) {
        points[count][i] = dist(eng);
        distance_sq_sum += points[count][i]*points[count][i];
      }
      if (distance_sq_sum <= 1.0) {
        #pragma omp atomic
        ++count;
        printf("Id: %d  count: %d\n", omp_get_thread_num(), count);
      } else {
        #pragma omp atomic
        ++ notincluded;
      }
    }
  }
  double end = omp_get_wtime();

  printf("sampling time: %f\n", end-start);
  printf("notincluded %d\n", notincluded);




  return 0;
}
