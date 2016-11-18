#ifndef KANN_MIN_H
#define KANN_MIN_H

#define KANN_MM_DEFAULT 0
#define KANN_MM_SGD     1
#define KANN_MM_RMSPROP 2

#define KANN_MB_DEFAULT 0
#define KANN_MB_CONST   1

typedef struct {
	int n, epoch;
	short mini_algo, batch_algo;
	float lr;
	float decay;
	float *maux, *baux;
} kann_min_t;

#ifdef __cplusplus
extern "C" {
#endif

kann_min_t *kann_min_init(int mini_algo, int batch_algo, int n);
void kann_min_destroy(kann_min_t *m);
void kann_min_mini_update(kann_min_t *m, const float *g, float *t);
void kann_min_batch_finish(kann_min_t *m, const float *t);

#ifdef __cplusplus
}
#endif

#endif
