#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "kann_rand.h"
#include "kann.h"

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

int kann_n_in(kann_t *a)
{
	return a->in == 0? -1 : a->in->n_d == 1? a->in->d[0] : kad_len(a->in) / a->in->d[0];
}

int kann_n_out(kann_t *a)
{
	return a->out_pre == 0? -1 : a->out_pre->n_d == 1? a->out_pre->d[0] : kad_len(a->out_pre) / a->out_pre->d[0];
}

int kann_n_par(kann_t *a)
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
	mo->lr = 0.001f;
	mo->fv = 0.1f;
	mo->mb_size = 64;
	mo->epoch_lazy = 10;
	mo->max_epoch = 100;
	mo->decay = 0.9f;
}

static void kann_set_batch_size(int B, int n_node, kad_node_t **node)
{
	int i;
	for (i = 0; i < n_node; ++i) {
		kad_node_t *p = node[i];
		if (p->n_child == 0 && (p->label == KAD_LABEL_IN || p->label == KAD_LABEL_OUT_PRE || p->label == KAD_LABEL_OUT_TRUTH))
			p->d[0] = B;
	}
	for (i = 0; i < n_node; ++i) {
		kad_node_t *p = node[i];
		if (p->n_child == 0) continue;
		kad_op_list[p->op](p, KAD_SYNC_DIM);
		kad_op_list[p->op](p, KAD_ALLOC);
		p->_.x = (float*)realloc(p->_.x, kad_len(p) * sizeof(float));
	}
}

void kann_train_fnn(const kann_mopt_t *mo, kann_t *a, int n, float **_x, float **_y) // TODO: hard coded to RMSprop for now
{
	extern void kann_RMSprop(int n, float h0, const float *h, float decay, const float *g, float *t, float *r);
	float **x, **y, **vx, **vy, *bx, *by, *rmsp_r;
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
	n_train = mo->fv > 0.0f && mo->fv < 1.0f? (int)(mo->fv * n + .499) : n;
	n_validate = n - n_train;
	vx = x + n_train;
	vy = y? y + n_train : 0;

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
		int rest = n_train;
		kann_shuffle(a->rng.data, n_train, x, y, 0);
		while (rest > 0) {
			int j, mb = rest < mo->mb_size? rest : mo->mb_size;
			kann_set_batch_size(mb, a->n, a->v);
//			kad_debug(stderr, a->n, a->v);
			for (j = 0; j < mb; ++j) {
				memcpy(&bx[j*n_in],  &x[rest+j], n_in  * sizeof(float));
				memcpy(&by[j*n_out], &y[rest+j], n_out * sizeof(float));
			}
			//fprintf(stderr, "here\n");
			kad_eval(a->n, a->v, 1);
			//fprintf(stderr, "there\n");
			kann_RMSprop(n_par, mo->lr, 0, mo->decay, a->g, a->t, rmsp_r);
			rest -= mb;
		}
		fprintf(stderr, "here: %g\n", a->v[a->n-1]->_.x[0]);
	}

	// free
	free(rmsp_r);
	free(by); free(bx);
	free(y); free(x);
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
