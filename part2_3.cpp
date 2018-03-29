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
const int N = 1'000'000;
alignas(32) static float points[N][NUM_DIMENSIONS];

int main() {
  uniform_real_distribution<float> dist(-1, 1);
  static int count = 0;
  double start = omp_get_wtime();

   #pragma omp parallel
    {
      default_random_engine eng(omp_get_thread_num());
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
        }
       }
    }

  double end = omp_get_wtime();
  printf("sampling time: %f\n", end-start);

  return 0;
}
