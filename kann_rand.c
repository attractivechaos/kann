#include <math.h>
#include <stdlib.h>
#include "kann_rand.h"

#define KR_NN 312
#define KR_MM 156
#define KR_UM 0xFFFFFFFF80000000ULL /* Most significant 33 bits */
#define KR_LM 0x7FFFFFFFULL /* Least significant 31 bits */

typedef struct {
	int mti, n_iset;
	double n_gset;
	uint64_t mt[KR_NN];
} krand_t;

static krand_t kr_aux;

static void kr_srand0(uint64_t seed, krand_t *kr)
{
	kr->mt[0] = seed;
	for (kr->mti = 1; kr->mti < KR_NN; ++kr->mti) 
		kr->mt[kr->mti] = 6364136223846793005ULL * (kr->mt[kr->mti - 1] ^ (kr->mt[kr->mti - 1] >> 62)) + kr->mti;
}

void *kann_srand_r(uint64_t seed)
{
	krand_t *kr;
	kr = (krand_t*)calloc(1, sizeof(krand_t));
	kr_srand0(seed, kr);
	return kr;
}

void kann_srand(uint64_t seed)
{
	kr_srand0(seed, &kr_aux);
}

// 64-bit Mersenne Twister pseudorandom number generator, originally written by Takuji Nishimura and Makoto Matsumoto
uint64_t kann_lrand(void *_kr)
{
	krand_t *kr = _kr? (krand_t*)_kr : &kr_aux;
	uint64_t x;
	static const uint64_t mag01[2] = { 0, 0xB5026F5AA96619E9ULL };
    if (kr->mti >= KR_NN) {
		int i;
		if (kr->mti == KR_NN + 1) kr_srand0(5489ULL, kr);
        for (i = 0; i < KR_NN - KR_MM; ++i) {
            x = (kr->mt[i] & KR_UM) | (kr->mt[i+1] & KR_LM);
            kr->mt[i] = kr->mt[i + KR_MM] ^ (x>>1) ^ mag01[(int)(x&1)];
        }
        for (; i < KR_NN - 1; ++i) {
            x = (kr->mt[i] & KR_UM) | (kr->mt[i+1] & KR_LM);
            kr->mt[i] = kr->mt[i + (KR_MM - KR_NN)] ^ (x>>1) ^ mag01[(int)(x&1)];
        }
        x = (kr->mt[KR_NN - 1] & KR_UM) | (kr->mt[0] & KR_LM);
        kr->mt[KR_NN - 1] = kr->mt[KR_MM - 1] ^ (x>>1) ^ mag01[(int)(x&1)];
        kr->mti = 0;
    }
    x = kr->mt[kr->mti++];
    x ^= (x >> 29) & 0x5555555555555555ULL;
    x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
    x ^= (x << 37) & 0xFFF7EEE000000000ULL;
    x ^= (x >> 43);
    return x;
}

double kann_drand(void *_kr)
{
	return (kann_lrand(_kr) >> 11) * (1.0/9007199254740992.0); // double-precision numbers have 53-bit
}

double kann_normal(void *_kr)
{ 
	krand_t *kr = _kr? (krand_t*)_kr : &kr_aux;
	if (kr->n_iset == 0) {
		double fac, rsq, v1, v2; 
		do { 
			v1 = 2.0 * kann_drand(kr) - 1.0;
			v2 = 2.0 * kann_drand(kr) - 1.0; 
			rsq = v1 * v1 + v2 * v2;
		} while (rsq >= 1.0 || rsq == 0.0);
		fac = sqrt(-2.0 * log(rsq) / rsq); 
		kr->n_gset = v1 * fac; 
		kr->n_iset = 1;
		return v2 * fac;
	} else {
		kr->n_iset = 0;
		return kr->n_gset;
	}
}

void kann_shuffle(void *kr, int n, float **x, float **y, char **rname)
{
   int i, *s;
   s = (int*)malloc(n * sizeof(int));
   for (i = n - 1; i >= 0; --i)
       s[i] = (int)(kann_drand(kr) * (i+1));
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
