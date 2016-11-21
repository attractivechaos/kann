#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "kann_rand.h"

const uint64_t kann_seed1 = 1181783497276652981ULL;

typedef struct {
	uint64_t s[2];
	double n_gset;
	int n_iset;
	volatile int lock;
} kann_rand_t;

static kann_rand_t kann_rng = { {11ULL, kann_seed1}, 0.0, 0, 0 };

static inline uint64_t xorshift128plus(uint64_t s[2])
{
	uint64_t x, y;
	x = s[0], y = s[1];
	s[0] = y;
	x ^= x << 23;
	s[1] = x ^ y ^ (x >> 17) ^ (y >> 26);
	y += s[1];
	return y;
}

void kann_srand(uint64_t seed0)
{
	kann_rand_t *r = &kann_rng;
	while (__sync_lock_test_and_set(&r->lock, 1));
	memset(r, 0, sizeof(kann_rand_t));
	r->s[0] = seed0, r->s[1] = kann_seed1;
	__sync_lock_release(&r->lock);
}

static inline uint64_t kann_rand_unsafe(kann_rand_t *r)
{
	return xorshift128plus(r->s);
}

static inline double kann_drand_unsafe(kann_rand_t *r)
{
	return (xorshift128plus(r->s)>>11) * (1.0/9007199254740992.0);
}

static double kann_normal_unsafe(kann_rand_t *r)
{
	if (r->n_iset == 0) {
		double fac, rsq, v1, v2; 
		do { 
			v1 = 2.0 * kann_drand_unsafe(r) - 1.0;
			v2 = 2.0 * kann_drand_unsafe(r) - 1.0; 
			rsq = v1 * v1 + v2 * v2;
		} while (rsq >= 1.0 || rsq == 0.0);
		fac = sqrt(-2.0 * log(rsq) / rsq); 
		r->n_gset = v1 * fac; 
		r->n_iset = 1;
		return v2 * fac;
	} else {
		r->n_iset = 0;
		return r->n_gset;
	}
}

uint64_t kann_rand(void)
{
	uint64_t x;
	kann_rand_t *r = &kann_rng;
	while (__sync_lock_test_and_set(&r->lock, 1));
	x = kann_rand_unsafe(r);
	__sync_lock_release(&r->lock);
	return x;
}

double kann_drand(void)
{
	double x;
	kann_rand_t *r = &kann_rng;
	while (__sync_lock_test_and_set(&r->lock, 1));
	x = kann_drand_unsafe(r);
	__sync_lock_release(&r->lock);
	return x;
}

double kann_normal(void)
{
	double x;
	kann_rand_t *r = &kann_rng;
	while (__sync_lock_test_and_set(&r->lock, 1));
	x = kann_normal_unsafe(r);
	__sync_lock_release(&r->lock);
	return x;
}

void kann_shuffle(int n, float **x, float **y, char **rname)
{
	int i, *s;
	kann_rand_t *r = &kann_rng;

	s = (int*)malloc(n * sizeof(int));
	while (__sync_lock_test_and_set(&r->lock, 1));
	for (i = n - 1; i >= 0; --i)
		s[i] = (int)(kann_drand_unsafe(r) * (i+1));
	__sync_lock_release(&r->lock);
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

void kann_rand_weight(int n_row, int n_col, float *w)
{
	int i, j;
	double s;
	kann_rand_t *r = &kann_rng;

	s = 1.0 / sqrt(n_col);
	while (__sync_lock_test_and_set(&r->lock, 1));
	for (i = 0; i < n_row; ++i)
		for (j = 0; j < n_col; ++j)
			w[i*n_col+j] = kann_normal_unsafe(r) * s;
	__sync_lock_release(&r->lock);
}
