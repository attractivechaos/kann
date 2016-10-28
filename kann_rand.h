#ifndef KANN_RAND_H
#define KANN_RAND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void kann_srand(uint64_t seed);
void *kann_srand_r(uint64_t seed); // remember to call free() on the returned pointer
uint64_t kann_lrand(void *kr);
double kann_drand(void *kr);
double kann_normal(void *kr);
void kann_shuffle(void *kr, int n, float **x, float **y, char **rname);

#ifdef __cplusplus
}
#endif

#endif
