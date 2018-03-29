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
alignas(32) static float points[N][NUM_DIMENSIONS];

int main() {

  uniform_real_distribution<float> dist(-1, 1);
  static int count = 0;
  static int threads_num = omp_get_num_threads();
  double start = omp_get_wtime();

  #pragma omp parallel
  {
    int nthread = omp_get_thread_num();
    default_random_engine eng(nthread);
    float distance_sq_sum;
    float temp[NUM_DIMENSIONS];
    int booked;

    while (count < N) {
      distance_sq_sum = 0.0;

  #pragma omp for
      for (int i = 0; i < NUM_DIMENSIONS; ++i) {
        temp[i] = dist(eng);
        distance_sq_sum += temp[i]*temp[i];
      }

      if (distance_sq_sum <= 1.0) {

  #pragma omp critical
  {
      booked = count;
      ++count;
  }

  #pragma omp for
        for (int i = 0; i < NUM_DIMENSIONS; ++i) {
          points[booked][i] = temp[i];
        }

        printf("Id: %d  count: %d\n", nthread, booked);
      }
    }
    printf("%d %s\n", nthread," Finished");
  }

  // #pragma omp for
  //   for (int i = nthread; i < N; i+=threads_num ) {
  //     distance_sq_sum = 2.0;
  //     while (distance_sq_sum > 1.0) {
  //       distance_sq_sum = 0.0;
  //       for (int j = 0; j < NUM_DIMENSIONS; ++j) {
  //         temp[j] = dist(eng);
  //         distance_sq_sum += temp[j]*temp[j];
  //       }
  //     }
  //     for (int k = 0; k < NUM_DIMENSIONS; ++k) {
  //       points[i][k] = temp[k];
  //     }
  //   }
// 
  // }
  double end = omp_get_wtime();
  printf("sampling time: %f\n", end-start);

  return 0;
}
