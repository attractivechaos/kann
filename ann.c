#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "kann_rand.h"
#include "kann.h"

#define KANN_MAGIC "KAN\1"

int kann_verbose = 3;

kann_t *kann_init(uint64_t seed)
{
	kann_t *a;
	a = (kann_t*)calloc(1, sizeof(kann_t));
	a->rng.data = kann_srand_r(seed);
	a->rng.func = kann_drand;
	return a;
}

void kann_destroy(kann_t *a)
{
	free(a->t); free(a->g);
	if (a->v) kad_free(a->n, a->v);
	free(a->rng.data);
	free(a);
}

void kann_sync_index(kann_t *a)
{
	int i;
	a->i_in = a->i_out = a->i_truth = a->i_cost = -1;
	for (i = 0; i < a->n; ++i) {
		switch (a->v[i]->label) {
			case KANN_LABEL_IN: a->i_in = i; break;
			case KANN_LABEL_OUT: a->i_out = i; break;
			case KANN_LABEL_TRUTH: a->i_truth = i; break;
			case KANN_LABEL_COST: a->i_cost = i; break;
		}
	}
}

void kann_collate_var(kann_t *a)
{
	int i, j, n_par;
	n_par = kann_n_par(a);
	a->t = (float*)realloc(a->t, n_par * sizeof(float));
	a->g = (float*)realloc(a->g, n_par * sizeof(float));
	memset(a->g, 0, n_par * sizeof(float));
	for (i = j = 0; i < a->n; ++i) {
		kad_node_t *v = a->v[i];
		if (v->n_child == 0 && v->to_back) {
			int l;
			l = kad_len(v);
			memcpy(&a->t[j], v->x, l * sizeof(float));
			free(v->x);
			v->x = &a->t[j];
			v->g = &a->g[j];
			j += l;
		}
	}
}

int kann_n_in(const kann_t *a)
{
	return a->i_in < 0? -1 : a->v[a->i_in]->n_d == 1? a->v[a->i_in]->d[0] : kad_len(a->v[a->i_in]) / a->v[a->i_in]->d[0];
}

int kann_n_out(const kann_t *a)
{
	return a->i_out < 0? -1 : a->v[a->i_out]->n_d == 1? a->v[a->i_out]->d[0] : kad_len(a->v[a->i_out]) / a->v[a->i_out]->d[0];
}

void kann_mopt_init(kann_mopt_t *mo)
{
	memset(mo, 0, sizeof(kann_mopt_t));
	mo->lr = 0.01f;
	mo->fv = 0.1f;
	mo->mb_size = 64;
	mo->epoch_lazy = 10;
	mo->max_epoch = 100;
	mo->decay = 0.9f;
}

static void kann_set_batch_size(kann_t *a, int B)
{
	int i;
	for (i = 0; i < a->n; ++i)
		if (a->v[i]->label == KANN_LABEL_IN || a->v[i]->label == KANN_LABEL_TRUTH)
			a->v[i]->d[0] = B;
	for (i = 0; i < a->n; ++i) {
		kad_node_t *p = a->v[i];
		if (p == 0 || p->n_child == 0) continue;
		kad_op_list[p->op](p, KAD_SYNC_DIM);
		kad_op_list[p->op](p, KAD_ALLOC);
		p->x = (float*)realloc(p->x, kad_len(p) * sizeof(float));
		p->g = (float*)realloc(p->g, kad_len(p) * sizeof(float));
	}
}

kann_t *kann_rnn_unroll(kann_t *a, int len, int pre_pool)
{
	int n, i, k;
	kad_node_t **v;
	kann_t *b;
	b = (kann_t*)calloc(1, sizeof(kann_t));
	b->rng = a->rng, b->t = a->t, b->g = a->g;
	if (pre_pool) {
	} else {
		kad_node_t **t;
		v = kad_unroll(a->n, a->v, len, &n);
		t = (kad_node_t**)calloc(len, sizeof(kad_node_t*));
		for (i = k = 0; i < n; ++i) {
			if (v[i]->label == KANN_LABEL_OUT) {
				t[k] = kad_par(0, 0);
				kad_sync_dim1(t[k], v[i]);
				++k;
			}
		}
	}
	return b;
}

void kann_train_fnn(const kann_mopt_t *mo, kann_t *a, int n, float **_x, float **_y) // TODO: hard coded to RMSprop for now
{
	extern void kann_RMSprop(int n, float h0, const float *h, float decay, const float *g, float *t, float *r);
	float **x, **y, *bx, *by, *rmsp_r;
	int i, n_train, n_validate, n_in, n_out, n_par;

	// copy and shuffle
	x = (float**)malloc(n * sizeof(float*));
	y = _y? (float**)malloc(n * sizeof(float*)) : 0;
	for (i = 0; i < n; ++i) {
		x[i] = _x[i];
		if (y) y[i] = _y[i];
	}
	kann_shuffle(a->rng.data, n, x, y, 0);

	// set validation set
	n_validate = mo->fv > 0.0f && mo->fv < 1.0f? (int)(mo->fv * n + .499) : 0;
	n_train = n - n_validate;

	// prepare mini-batch buffer
	n_in = kann_n_in(a);
	n_out = kann_n_out(a);
	n_par = kann_n_par(a);
	bx = (float*)malloc(mo->mb_size * n_in * sizeof(float));
	by = (float*)malloc(mo->mb_size * n_out * sizeof(float));
	for (i = 0; i < a->n; ++i) {
		kad_node_t *p = a->v[i];
		if (p->n_child) continue;
		if (p->label == KANN_LABEL_IN) p->x = bx;
		else if (p->label == KANN_LABEL_TRUTH) p->x = by;
	}
	rmsp_r = (float*)calloc(n_par, sizeof(float));

	// main loop
	for (i = 0; i < mo->max_epoch; ++i) {
		int n_proc = 0;
		double running_cost = 0.0, val_cost = 0.0;
		kann_shuffle(a->rng.data, n_train, x, y, 0);
		while (n_proc < n_train) {
			int j, mb = n_train - n_proc < mo->mb_size? n_train - n_proc : mo->mb_size;
			kann_set_batch_size(a, mb);
			for (j = 0; j < mb; ++j) {
				memcpy(&bx[j*n_in],  x[n_proc+j], n_in  * sizeof(float));
				memcpy(&by[j*n_out], y[n_proc+j], n_out * sizeof(float));
			}
			running_cost += *kad_eval(a->n, a->v, a->i_cost) * mb;
			kad_grad(a->n, a->v, a->i_cost);
			kann_RMSprop(n_par, mo->lr, 0, mo->decay, a->g, a->t, rmsp_r);
			n_proc += mb;
		}
		n_proc = 0;
		while (n_proc < n_validate) {
			int j, mb = n_validate - n_proc < mo->mb_size? n_validate - n_proc : mo->mb_size;
			kann_set_batch_size(a, mb);
			for (j = 0; j < mb; ++j) {
				memcpy(&bx[j*n_in],  x[n_proc+j], n_in  * sizeof(float));
				memcpy(&by[j*n_out], y[n_proc+j], n_out * sizeof(float));
			}
			val_cost += *kad_eval(a->n, a->v, a->i_cost) * mb;
			n_proc += mb;
		}
		if (kann_verbose >= 3) {
			if (n_validate == 0) fprintf(stderr, "running cost: %g\n", running_cost / n_train);
			else fprintf(stderr, "running cost: %g; validation cost: %g\n", running_cost / n_train, val_cost / n_validate);
		}
	}

	// free
	free(rmsp_r);
	free(by); free(bx);
	free(y); free(x);
}

const float *kann_apply_fnn1(kann_t *a, float *x)
{
	kann_set_batch_size(a, 1);
	a->v[a->i_in]->x = x;
	kad_eval(a->n, a->v, a->i_out);
	return a->v[a->i_out]->x;
}

/*************
 * Model I/O *
 *************/

void kann_write(const char *fn, const kann_t *ann)
{
	FILE *fp;
	fp = fn && strcmp(fn, "-")? fopen(fn, "wb") : stdout;
	fwrite(KANN_MAGIC, 1, 4, fp);
	kad_write(fp, ann->n, ann->v);
	fwrite(ann->t, sizeof(float), kann_n_par(ann), fp);
	fclose(fp);
}

kann_t *kann_read(const char *fn)
{
	FILE *fp;
	char magic[4];
	kann_t *ann;
	int i, j, n_par;

	fp = fn && strcmp(fn, "-")? fopen(fn, "rb") : stdin;
	fread(magic, 1, 4, fp);
	if (strncmp(magic, KANN_MAGIC, 4) != 0) {
		fclose(fp);
		return 0;
	}
	ann = (kann_t*)calloc(1, sizeof(kann_t));
	ann->rng.data = kann_srand_r(11);
	ann->rng.func = kann_drand;
	ann->v = kad_read(fp, &ann->n);
	kann_sync_index(ann);
	n_par = kann_n_par(ann);
	ann->t = (float*)malloc(n_par * sizeof(float));
	ann->g = (float*)calloc(n_par, sizeof(float));
	fread(ann->t, sizeof(float), n_par, fp);
	fclose(fp);

	for (i = j = 0; i < ann->n; ++i) {
		kad_node_t *p = ann->v[i];
		if (p->n_child == 0 && p->to_back) {
			p->x = &ann->t[j];
			p->g = &ann->g[j];
			j += kad_len(p);
		}
	}
	assert(j == n_par);
	return ann;
}

/**************
 * Optimizers *
 **************/

#ifdef __SSE__
#include <xmmintrin.h>

void kann_RMSprop(int n, float h0, const float *h, float decay, const float *g, float *t, float *r)
{
	int i, n4 = n>>2<<2;
	__m128 vh, vg, vr, vt, vd, vd1, tmp, vtiny;
	vh = _mm_set1_ps(h0);
	vd = _mm_set1_ps(decay);
	vd1 = _mm_set1_ps(1.0f - decay);
	vtiny = _mm_set1_ps(1e-6f);
	for (i = 0; i < n4; i += 4) {
		vt = _mm_loadu_ps(&t[i]);
		vr = _mm_loadu_ps(&r[i]);
		vg = _mm_loadu_ps(&g[i]);
		if (h) vh = _mm_loadu_ps(&h[i]);
		vr = _mm_add_ps(_mm_mul_ps(vd1, _mm_mul_ps(vg, vg)), _mm_mul_ps(vd, vr));
		_mm_storeu_ps(&r[i], vr);
		tmp = _mm_sub_ps(vt, _mm_mul_ps(_mm_mul_ps(vh, _mm_rsqrt_ps(_mm_add_ps(vtiny, vr))), vg));
		_mm_storeu_ps(&t[i], tmp);
	}
	for (; i < n; ++i) {
		r[i] = (1. - decay) * g[i] * g[i] + decay * r[i];
		t[i] -= (h? h[i] : h0) / sqrt(1e-6 + r[i]) * g[i];
	}
}
#else
void kann_RMSprop(int n, float h0, const float *h, float decay, const float *g, float *t, float *r)
{
	int i;
	for (i = 0; i < n; ++i) {
		float lr = h? h[i] : h0;
		r[i] = (1. - decay) * g[i] * g[i] + decay * r[i];
		t[i] -= lr / sqrt(1e-6 + r[i]) * g[i];
	}
}
#endif
