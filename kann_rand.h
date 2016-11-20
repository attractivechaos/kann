#ifndef KANN_RAND_H
#define KANN_RAND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void kann_srand(uint64_t seed);
double kann_drand(void);
double kann_normal(void);
void kann_shuffle(int n, float **x, float **y, char **rname);
void kann_rand_weight(int n_row, int n_col, float *w);

#ifdef __cplusplus
}
#endif

#endif
