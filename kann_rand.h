#ifndef KANN_RAND_H
#define KANN_RAND_H

#include <stdint.h>

//#define KANN_RAND_LOCK

#ifdef __cplusplus
extern "C" {
#endif

void kann_srand(uint64_t seed);
double kann_drand(void);
double kann_normal(int *iset, double *gset);
void kann_shuffle(int n, float **x, float **y, char **rname);

#ifdef __cplusplus
}
#endif

#endif
