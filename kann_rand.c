#include <math.h>
#include <stdlib.h>
#include "kann_rand.h"

#define KANN_RNG_INIT 1181783497276652981ULL

static uint64_t kann_rng[2] = { 11ULL, KANN_RNG_INIT };

#ifdef KANN_RAND_LOCK
static volatile int kann_rng_lock = 0;
#endif

static inline uint64_t xorshift128plus(uint64_t s[2])
{
	uint64_t x, y;
#ifdef KANN_RAND_LOCK
	while (__sync_lock_test_and_set(&kann_rng_lock, 1)) while (kann_rng_lock); // a spin lock
#endif
	x = s[0], y = s[1];
	s[0] = y;
	x ^= x << 23;
	s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
	y += s[1];
#ifdef KANN_RAND_LOCK
	__sync_lock_release(&kann_rng_lock);
#endif
	return y;
}

void kann_srand(uint64_t seed)
{
	kann_rng[0] = seed, kann_rng[1] = KANN_RNG_INIT;
}

double kann_drand(void)
{
	return (xorshift128plus(kann_rng)>>11) * (1.0/9007199254740992.0);
}

double kann_normal(int *iset, double *gset)
{ 
	if (*iset == 0) {
		double fac, rsq, v1, v2; 
		do { 
			v1 = 2.0 * kann_drand() - 1.0;
			v2 = 2.0 * kann_drand() - 1.0; 
			rsq = v1 * v1 + v2 * v2;
		} while (rsq >= 1.0 || rsq == 0.0);
		fac = sqrt(-2.0 * log(rsq) / rsq); 
		*gset = v1 * fac; 
		*iset = 1;
		return v2 * fac;
	} else {
		*iset = 0;
		return *gset;
	}
}

void kann_shuffle(int n, float **x, float **y, char **rname)
{
	int i, *s;
	s = (int*)malloc(n * sizeof(int));
	for (i = n - 1; i >= 0; --i)
		s[i] = (int)(kann_drand() * (i+1));
	for (i = n - 1; i >= 0; --i) {
		float *tf;
		char *ts;
		int j = s[i];
		if (x) tf = x[i], x[i] = x[j], x[j] = tf;
		if (y) tf = y[i], y[i] = y[j], y[j] = tf;
		if (rname) ts = rname[i], rname[i] = rname[j], rname[j] = ts;
	}
	free(s);
}
