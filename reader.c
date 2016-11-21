#include <assert.h>
#include <stdlib.h>
#include "kann_rand.h"
#include "kann.h"

typedef struct {
	int n_t, n_v, d_x, d_y;
	int n_proc_t, n_proc_v;
	float **x, **y;
} rdr_xy_t;

void *kann_rdr_xy_new(int n, float frac_validate, int d_x, float **x, int d_y, float **y)
{
	rdr_xy_t *d;
	int i;
	d = (rdr_xy_t*)calloc(1, sizeof(rdr_xy_t));
	d->d_x = d_x, d->d_y = d_y;
	d->n_v = (int)(n * frac_validate + .499);
	d->n_t = n - d->n_v;
	d->x = (float**)malloc(n * sizeof(float*));
	if (y) d->y = (float**)malloc(n * sizeof(float*));
	for (i = 0; i < n; ++i) {
		d->x[i] = x[i];
		if (y) d->y[i] = y[i];
	}
	kann_shuffle(n, d->x, d->y, 0);
	return d;
}

void kann_rdr_xy_delete(void *data)
{
	rdr_xy_t *d = (rdr_xy_t*)data;
	free(d->x); free(d->y); free(d);
}

int kann_rdr_xy_read(void *data, int action, int *len, int max_bs, float **x, float **y)
{
	rdr_xy_t *d = (rdr_xy_t*)data;
	int i, bs = 0;

	if (len) *len = 1;
	if (action == KANN_RA_RESET) {
		d->n_proc_t = d->n_proc_v = 0;
		kann_shuffle(d->n_t, d->x, d->y, 0);
	} else if (action == KANN_RA_READ_TRAIN) {
		bs = d->n_proc_t + max_bs < d->n_t? max_bs : d->n_t - d->n_proc_t;
		if (bs > 0) {
			for (i = 0; i < bs; ++i) {
				memcpy(&x[0][i * d->d_x], d->x[d->n_proc_t + i], d->d_x * sizeof(float));
				if (d->y && y) memcpy(&y[0][i * d->d_y], d->y[d->n_proc_t + i], d->d_y * sizeof(float));
			}
			d->n_proc_t += bs;
		}
	} else if (action == KANN_RA_READ_VALIDATE) {
		bs = d->n_proc_v + max_bs < d->n_v? max_bs : d->n_v - d->n_proc_v;
		if (bs > 0) {
			for (i = 0; i < bs; ++i) {
				memcpy(&x[0][i * d->d_x], d->x[d->n_t + d->n_proc_v + i], d->d_x * sizeof(float));
				if (d->y && y) memcpy(&y[0][i * d->d_y], d->y[d->n_t + d->n_proc_v + i], d->d_y * sizeof(float));
			}
			d->n_proc_v += bs;
		}
	}
	return bs;
}
