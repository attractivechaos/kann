#include <assert.h>
#include <stdlib.h>
#include "kann.h"

typedef struct {
	int n[2], n_proc[2];
	int d_x, d_y;
	float **x, **y;
} rdr_xy_t;

void *kann_rdr_xy_new(int n, float frac_validate, int d_x, float **x, int d_y, float **y)
{
	rdr_xy_t *d;
	int i;
	d = (rdr_xy_t*)calloc(1, sizeof(rdr_xy_t));
	d->d_x = d_x, d->d_y = d_y;
	d->n[1] = (int)(n * frac_validate + .499);
	d->n[0] = n - d->n[1];
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

int kann_rdr_xy_read(void *data, int action, int max_len, float *x, float *y)
{
	rdr_xy_t *d = (rdr_xy_t*)data;
	if (action == KANN_RDR_BATCH_RESET) {
		d->n_proc[0] = d->n_proc[1] = 0;
		kann_shuffle(d->n[0], d->x, d->y, 0);
	} else if (action == KANN_RDR_READ_TRAIN || action == KANN_RDR_READ_VALIDATE) {
		int k = action == KANN_RDR_READ_TRAIN? 0 : 1, shift = k? d->n[0] : 0;
		if (d->n_proc[k] < d->n[k]) {
			memcpy(x, d->x[d->n_proc[k]+shift], d->d_x * sizeof(float));
			if (d->y && y) memcpy(y, d->y[d->n_proc[k]+shift], d->d_y * sizeof(float));
			++d->n_proc[k];
		} else return 0;
	}
	return 1;
}

void kann_fnn_train(const kann_mopt_t *mo, kann_t *a, int n, float **x, float **y)
{
	void *data;
	data = kann_rdr_xy_new(n, mo->fv, kann_n_in(a), x, kann_n_out(a), y);
	kann_train(mo, a, kann_rdr_xy_read, data);
	kann_rdr_xy_delete(data);
}
