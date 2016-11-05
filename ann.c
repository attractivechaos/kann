#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "kann_rand.h"
#include "kann.h"

#define KANN_MAGIC "KAN\1"

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
	kad_free_node(a->out_est);
	free(a->rng.data);
	free(a);
}

void kann_sync(kann_t *a)
{
	int i;
	a->in = a->out_pre = a->out_truth = 0;
	for (i = 0; i < a->n; ++i) {
		kad_node_t *v = a->v[i];
		switch (v->label) {
			case KAD_LABEL_IN: a->in = v; break;
			case KAD_LABEL_OUT_PRE: a->out_pre = v; break;
			case KAD_LABEL_OUT_TRUTH: a->out_truth = v; break;
		}
	}
}

int kann_n_in(const kann_t *a)
{
	return a->in == 0? -1 : a->in->n_d == 1? a->in->d[0] : kad_len(a->in) / a->in->d[0];
}

int kann_n_out(const kann_t *a)
{
	return a->out_pre == 0? -1 : a->out_pre->n_d == 1? a->out_pre->d[0] : kad_len(a->out_pre) / a->out_pre->d[0];
}

int kann_n_par(const kann_t *a)
{
	int i, n = 0;
	for (i = 0; i < a->n; ++i)
		if (a->v[i]->op == 0 && a->v[i]->to_back)
			n += kad_len(a->v[i]);
	return n;
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

static void kann_set_batch_size(int B, int n_node, kad_node_t **node, kad_node_t *extra)
{
	int i;
	for (i = 0; i < n_node; ++i) {
		kad_node_t *p = node[i];
		if (p->n_child == 0 && (p->label == KAD_LABEL_IN || p->label == KAD_LABEL_OUT_PRE || p->label == KAD_LABEL_OUT_TRUTH))
			p->d[0] = B;
	}
	for (i = 0; i <= n_node; ++i) {
		kad_node_t *p = i < n_node? node[i] : extra;
		if (p == 0 || p->n_child == 0) continue;
		kad_op_list[p->op](p, KAD_SYNC_DIM);
		kad_op_list[p->op](p, KAD_ALLOC);
		p->_.x = (float*)realloc(p->_.x, kad_len(p) * sizeof(float));
		p->g = (float*)realloc(p->g, kad_len(p) * sizeof(float));
	}
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
		if (p->label == KAD_LABEL_IN) p->_.cx = bx;
		else if (p->label == KAD_LABEL_OUT_TRUTH) p->_.cx = by;
	}
	rmsp_r = (float*)calloc(n_par, sizeof(float));

	// main loop
	for (i = 0; i < mo->max_epoch; ++i) {
		int n_proc = 0;
		double running_cost = 0.0, val_cost = 0.0;
		kann_shuffle(a->rng.data, n_train, x, y, 0);
		while (n_proc < n_train) {
			int j, mb = n_train - n_proc < mo->mb_size? n_train - n_proc : mo->mb_size;
			kann_set_batch_size(mb, a->n, a->v, a->out_est);
			for (j = 0; j < mb; ++j) {
				memcpy(&bx[j*n_in],  x[n_proc+j], n_in  * sizeof(float));
				memcpy(&by[j*n_out], y[n_proc+j], n_out * sizeof(float));
			}
			running_cost += kad_eval(a->n, a->v, 1) * mb;
			kann_RMSprop(n_par, mo->lr, 0, mo->decay, a->g, a->t, rmsp_r);
			n_proc += mb;
		}
		n_proc = 0;
		while (n_proc < n_validate) {
			int j, mb = n_validate - n_proc < mo->mb_size? n_validate - n_proc : mo->mb_size;
			kann_set_batch_size(mb, a->n, a->v, a->out_est);
			for (j = 0; j < mb; ++j) {
				memcpy(&bx[j*n_in],  x[n_proc+j], n_in  * sizeof(float));
				memcpy(&by[j*n_out], y[n_proc+j], n_out * sizeof(float));
			}
			val_cost += kad_eval(a->n, a->v, 0) * mb;
			n_proc += mb;
		}
		fprintf(stderr, "running cost: %g; validation cost: %g\n", running_cost / n_train, val_cost / n_validate);
	}

	// free
	free(rmsp_r);
	free(by); free(bx);
	free(y); free(x);
}

void kann_apply_fnn(kann_t *a, int n, float **x)
{
	float *bx;
	int i, n_in;
	n_in = kann_n_in(a);
	kann_set_batch_size(n, a->n, a->v, a->out_est);
	bx = (float*)malloc(n * n_in * sizeof(float));
	for (i = 0; i < n; ++i)
		memcpy(&bx[i*n_in], x[i], n_in  * sizeof(float));
	a->in->_.cx = bx;
	kad_eval(a->n, a->v, 0);
	kad_for1(a->out_est);
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
	kad_write1(fp, ann->out_est);
	fwrite(ann->t, sizeof(float), kann_n_par(ann), fp);
	fclose(fp);
}

kann_t *kann_read(const char *fn)
{
	FILE *fp;
	char magic[4];
	kann_t *ann;
	int n_par;

	fp = fn && strcmp(fn, "-")? fopen(fn, "rb") : stdin;
	fread(magic, 1, 4, fp);
	if (strncmp(magic, KANN_MAGIC, 4) != 0) {
		fclose(fp);
		return 0;
	}
	ann = (kann_t*)calloc(1, sizeof(kann_t));
	ann->v = kad_read(fp, &ann->n);
	kann_sync(ann);
	n_par = kann_n_par(ann);
	ann->out_est = kad_read1(fp, 0);
	ann->out_est->child[0].p = ann->out_pre;
	ann->t = (float*)malloc(n_par * sizeof(float));
	ann->g = (float*)calloc(n_par, sizeof(float));
	fread(ann->t, sizeof(float), n_par, fp);
	fclose(fp);
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
