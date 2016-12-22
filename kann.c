#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "kann.h"

int kann_verbose = 3;

/*
  @@RNG: pseudo-random number generator
  @@LAYER: layers and model generation
  @@MISC: miscellaneous routines
  @@MIN: minimizer
  @@TRAIN: model training
  @@APPLY: applying a trained model
  @@TRAIN_FNN: simpler API for training FNN
  @@MODEL_IO: model I/O
*/

/***********************************************
 *** @@RNG: pseudo-random number generator ***
 ***********************************************/

#define KANN_SEED1 1181783497276652981ULL

typedef struct {
	uint64_t s[2];
	double n_gset;
	int n_iset;
	volatile int lock;
} kann_rand_t;

static kann_rand_t kann_rng = { {11ULL, KANN_SEED1}, 0.0, 0, 0 };

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
	r->s[0] = seed0, r->s[1] = KANN_SEED1;
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

void kann_normal_array(float sigma, int n, float *x)
{
	int i;
	double s = 1.0 / sigma;
	kann_rand_t *r = &kann_rng;
	while (__sync_lock_test_and_set(&r->lock, 1));
	for (i = 0; i < n; ++i) x[i] = kann_normal_unsafe(r) * s;
	__sync_lock_release(&r->lock);
}

/**********************************************
 *** @@LAYER: layers and model generation ***
 **********************************************/

kad_node_t *kann_new_weight(int n_row, int n_col)
{
	kad_node_t *w;
	w = kad_var(0, 0, 2, n_row, n_col);
	w->x = (float*)malloc(n_row * n_col * sizeof(float));
	kann_normal_array(sqrt(n_col), n_row * n_col, w->x);
	return w;
}

kad_node_t *kann_new_bias(int n)
{
	kad_node_t *b;
	b = kad_var(0, 0, 1, n);
	b->x = (float*)calloc(n, sizeof(float));
	return b;
}

kad_node_t *kann_new_weight_conv2d(int n_out, int n_in, int k_row, int k_col)
{
	kad_node_t *w;
	w = kad_var(0, 0, 4, n_out, n_in, k_row, k_col);
	w->x = (float*)malloc(kad_len(w) * sizeof(float));
	kann_normal_array(sqrt(n_in * k_row * k_col), n_out * n_in * k_row * k_col, w->x);
	return w;
}

kad_node_t *kann_new_weight_conv1d(int n_out, int n_in, int kernel_len)
{
	kad_node_t *w;
	w = kad_var(0, 0, 3, n_out, n_in, kernel_len);
	w->x = (float*)malloc(kad_len(w) * sizeof(float));
	kann_normal_array(sqrt(n_in * kernel_len), n_out * n_in * kernel_len, w->x);
	return w;
}

kad_node_t *kann_layer_input(int n1)
{
	kad_node_t *t;
	t = kad_feed(0, 2, 1, n1), t->ext_flag |= KANN_F_IN;
	return t;
}

kad_node_t *kann_layer_linear(kad_node_t *in, int n1)
{
	int n0;
	kad_node_t *w, *b;
	n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
	w = kann_new_weight(n1, n0);
	b = kann_new_bias(n1);
	return kad_add(kad_cmul(in, w), b);
}

kad_node_t *kann_layer_dropout(kad_node_t *t, float r)
{
	kad_node_t *s;
	s = kad_const(0, 0), s->ext_flag |= KANN_F_DROPOUT;
	s->x = (float*)calloc(1, sizeof(float));
	*s->x = r;
	return kad_dropout(t, s);
}

kad_node_t *kann_layer_rnn(kad_node_t *in, int n1, kad_node_t *(*af)(kad_node_t*))
{
	int n0;
	kad_node_t *h0, *w, *u, *b, *out;
	n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
	h0 = kad_var(0, 0, 2, 1, n1);
	h0->x = (float*)calloc(n1, sizeof(float));
	w = kann_new_weight(n1, n0);
	u = kann_new_weight(n1, n1);
	b = kann_new_bias(n1);
	out = af(kad_add(kad_add(kad_cmul(in, w), kad_cmul(h0, u)), b));
	out->pre = h0;
	return out;
}

kad_node_t *kann_layer_lstm(kad_node_t *in, int n1)
{
	int j, n0;
	kad_node_t *i, *f, *o, *g, *w, *u, *b, *h0, *c0, *c, *out;

	n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
	h0 = kad_var(0, 0, 2, 1, n1);
	h0->x = (float*)calloc(n1, sizeof(float));
	c0 = kad_var(0, 0, 2, 1, n1);
	c0->x = (float*)calloc(n1, sizeof(float));

	// i = sigm(x_t * W_i + h_{t-1} * U_i + b_i)
	w = kann_new_weight(n1, n0);
	u = kann_new_weight(n1, n1);
	b = kann_new_bias(n1);
	i = kad_sigm(kad_add(kad_add(kad_cmul(in, w), kad_cmul(h0, u)), b));
	// f = sigm(x_t * W_f + h_{t-1} * U_f + b_f)
	w = kann_new_weight(n1, n0);
	u = kann_new_weight(n1, n1);
	b = kann_new_bias(n1);
	for (j = 0; j < n1; ++j) b->x[j] = 1.0f; // see Jozefowicz et al on using a large bias
	f = kad_sigm(kad_add(kad_add(kad_cmul(in, w), kad_cmul(h0, u)), b));
	// o = sigm(x_t * W_o + h_{t-1} * U_o + b_o)
	w = kann_new_weight(n1, n0);
	u = kann_new_weight(n1, n1);
	b = kann_new_bias(n1);
	o = kad_sigm(kad_add(kad_add(kad_cmul(in, w), kad_cmul(h0, u)), b));
	// g = tanh(x_t * W_g + h_{t-1} * U_g + b_g)
	w = kann_new_weight(n1, n0);
	u = kann_new_weight(n1, n1);
	b = kann_new_bias(n1);
	g = kad_tanh(kad_add(kad_add(kad_cmul(in, w), kad_cmul(h0, u)), b));
	// c_t = c_{t-1} # f + g # i
	c = kad_add(kad_mul(f, c0), kad_mul(g, i)); // can't be kad_mul(c0, f)!!!
	c->pre = c0;
	// h_t = tanh(c_t) # o
	out = kad_mul(kad_tanh(c), o);
	out->pre = h0;
	return out;
}

kad_node_t *kann_layer_gru(kad_node_t *in, int n1)
{
	int n0;
	kad_node_t *r, *z, *w, *u, *b, *s, *h0, *out;

	n0 = in->n_d >= 2? kad_len(in) / in->d[0] : kad_len(in);
	h0 = kad_var(0, 0, 2, 1, n1);
	h0->x = (float*)calloc(n1, sizeof(float));

	// z = sigm(x_t * W_z + h_{t-1} * U_z + b_z)
	w = kann_new_weight(n1, n0);
	u = kann_new_weight(n1, n1);
	b = kann_new_bias(n1);
	z = kad_sigm(kad_add(kad_add(kad_cmul(in, w), kad_cmul(h0, u)), b));
	// r = sigm(x_t * W_r + h_{t-1} * U_r + b_r)
	w = kann_new_weight(n1, n0);
	u = kann_new_weight(n1, n1);
	b = kann_new_bias(n1);
	r = kad_sigm(kad_add(kad_add(kad_cmul(in, w), kad_cmul(h0, u)), b));
	// s = tanh(x_t * W_s + (h_{t-1} # r) * U_s + b_s)
	w = kann_new_weight(n1, n0);
	u = kann_new_weight(n1, n1);
	b = kann_new_bias(n1);
	s = kad_tanh(kad_add(kad_add(kad_cmul(in, w), kad_cmul(kad_mul(r, h0), u)), b)); // can't be kad_mul(h0, r)!!!
	// h_t = z # h_{t-1} + (1 - z) # s
	out = kad_add(kad_mul(kad_1minus(z), s), kad_mul(z, h0));
	out->pre = h0;
	return out;
}

kad_node_t *kann_layer_conv2d(kad_node_t *in, int n_flt, int k_rows, int k_cols, int stride, int pad)
{
	kad_node_t *w;
	w = kann_new_weight_conv2d(n_flt, in->d[1], k_rows, k_cols);
	return kad_conv2d(in, w, stride, stride, pad, pad);
}

kann_t *kann_gen(kad_node_t *cost)
{
	kann_t *a;
	int i, has_pivot = 0, has_recur = 0;

	if (cost->n_d != 0) return 0;
	cost->ext_flag |= KANN_F_COST;
	a = (kann_t*)calloc(1, sizeof(kann_t));
	a->v = kad_compile(&a->n, 1, cost);
	for (i = 0; i < a->n; ++i) {
		if (a->v[i]->pre) has_recur = 1;
		if (kad_is_pivot(a->v[i])) has_pivot = 1;
	}
	if (has_recur && !has_pivot) { // an RNN that doesn't have a pivot; then add a pivot on top of cost
		cost->ext_flag &= ~KANN_F_COST;
		cost = kad_avg(1, &cost), cost->ext_flag |= KANN_F_COST;
		free(a->v);
		a->v = kad_compile(&a->n, 1, cost);
	}
	kad_ext_collate(a->n, a->v, &a->x, &a->g, &a->c);
	return a;
}

kann_t *kann_layer_final(kad_node_t *t, int n_out, int type)
{
	kad_node_t *cost = 0, *truth = 0;
	assert(type == KANN_C_CEB || type == KANN_C_CEM);
	kad_drand = kann_drand;
	truth = kad_feed(0, 2, 1, n_out), truth->ext_flag |= KANN_F_TRUTH;
	t = kann_layer_linear(t, n_out);
	if (type == KANN_C_CEB) {
		t = kad_sigm(t);
		cost = kad_ce_bin(t, truth);
	} else if (type == KANN_C_CEM) {
		kad_node_t *t1;
		t1 = kad_const(0, 0), t1->ext_flag |= KANN_F_TEMP;
		t1->x = (float*)calloc(1, sizeof(float));
		*t1->x = 1.0f;
		t = kad_softmax(kad_mul(t, t1));
		cost = kad_ce_multi(t, truth);
	}
	t->ext_flag |= KANN_F_OUT, cost->ext_flag |= KANN_F_COST;
	return kann_gen(cost);
}

kann_t *kann_rnn_unroll(kann_t *a, int len)
{
	kann_t *b;
	b = (kann_t*)calloc(1, sizeof(kann_t));
	b->x = a->x, b->g = a->g, b->c = a->c; // these arrays are shared
	b->v = kad_unroll(a->n, a->v, len, &b->n);
	return b;
}

/**************************************
 *** @@MISC: miscellaneous routines ***
 **************************************/

void kann_delete_unrolled(kann_t *a)
{
	if (a && a->v) kad_delete(a->n, a->v);
	free(a);
}

void kann_delete(kann_t *a)
{
	free(a->x); free(a->g); free(a->c);
	kann_delete_unrolled(a);
}

void kann_set_by_flag(kann_t *a, int flag, float z)
{
	int i;
	for (i = 0; i < a->n; ++i)
		if ((a->v[i]->ext_flag & flag) && a->v[i]->n_d == 0)
			*a->v[i]->x = z;
}

static void kann_set_batch_size(kann_t *a, int B)
{
	int i;
	for (i = 0; i < a->n; ++i)
		if ((a->v[i]->ext_flag & (KANN_F_IN|KANN_F_TRUTH)) && (a->v[i]->d[0] != B || a->v[i]->x == 0))
			break;
	if (i == a->n) return; // no need to realloc
	for (i = 0; i < a->n; ++i)
		if (a->v[i]->ext_flag & (KANN_F_IN|KANN_F_TRUTH))
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

int kann_bind_by_flag(kann_t *a, int flag, float **x)
{
	int i, k;
	for (i = k = 0; i < a->n; ++i)
		if (kad_is_feed(a->v[i]) && (a->v[i]->ext_flag & flag))
			a->v[i]->x = x[k++];
	return k;
}

static inline int kann_n_by_flag(const kann_t *a, int flag)
{
	int i, n = 0;
	for (i = 0; i < a->n; ++i)
		if (a->v[i]->ext_flag & flag)
			n += a->v[i]->n_d > 1? kad_len(a->v[i]) / a->v[i]->d[0] : 1; // the first dimension is batch size
	return n;
}

int kann_n_in(const kann_t *a) { return kann_n_by_flag(a, KANN_F_IN); }
int kann_n_out(const kann_t *a) { return kann_n_by_flag(a, KANN_F_OUT); }

int kann_is_rnn(const kann_t *a)
{
	int i;
	for (i = 0; i < a->n; ++i)
		if (a->v[i]->pre) return 1;
	return 0;
}

/************************
 *** @@MIN: minimizer ***
 ************************/

typedef struct {
	int n, epoch;
	short mini_algo, batch_algo;
	float decay;
	// for Rprop-
	float rprop_dec, rprop_inc;
	float h_min, h_max;
	// persistent arrays
	float *maux, *baux, *h;
} kann_min_t;

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

kann_min_t *kann_min_new(int mini_algo, int batch_algo, int n)
{
	int i;
	kann_min_t *m;
	if (mini_algo <= 0) mini_algo = KANN_MM_RMSPROP;
	if (batch_algo <= 0) batch_algo = KANN_MB_iRprop;
	m = (kann_min_t*)calloc(1, sizeof(kann_min_t));
	m->mini_algo = mini_algo, m->batch_algo = batch_algo, m->n = n;
	m->h_min = 1e-7f, m->h_max = 10.0f;
	m->rprop_dec = 0.5f, m->rprop_inc = 1.2f;
	m->decay = 0.9f;
	m->h = (float*)calloc(n, sizeof(float));
	for (i = 0; i < n; ++i) m->h[i] = 0.001f;
	if (mini_algo == KANN_MM_RMSPROP) {
		m->maux = (float*)calloc(n, sizeof(float));
	}
	if (batch_algo == KANN_MB_iRprop) {
		m->baux = (float*)calloc(2 * n, sizeof(float));
	}
	return m;
}

void kann_min_set_lr(kann_min_t *m, float lr)
{
	int i;
	for (i = 0; i < m->n; ++i) m->h[i] = lr;
}

void kann_min_delete(kann_min_t *m)
{
	free(m->maux); free(m->baux); free(m->h); free(m);
}

void kann_min_mini_update(kann_min_t *m, const float *g, float *t)
{
	if (m->mini_algo == KANN_MM_RMSPROP) {
		kann_RMSprop(m->n, 0.0f, m->h, m->decay, g, t, m->maux);
	}
}

void kann_min_batch_finish(kann_min_t *m, const float *t)
{
	if (m->batch_algo == KANN_MB_iRprop) {
		int i;
		float *t0 = m->baux, *g0 = m->baux + m->n;
		if (m->epoch == 1) {
			for (i = 0; i < m->n; ++i) g0[i] = t[i] - t0[i];
		} else if (m->epoch > 1) {
			for (i = 0; i < m->n; ++i) {
				float g = t[i] - t0[i], tmp = g * g0[i];
				if (tmp > 0.0f) {
					m->h[i] *= m->rprop_inc;
					if (m->h[i] > m->h_max) m->h[i] = m->h_max;
				} else if (tmp < 0.0f) {
					m->h[i] *= m->rprop_dec;
					if (m->h[i] < m->h_min) m->h[i] = m->h_min;
					g = 0.0f;
				}
				g0[i] = g;
			}
		}
		memcpy(m->baux, t, m->n * sizeof(float));
	}
	++m->epoch;
}

/*******************************
 *** @@TRAIN: model training ***
 *******************************/

void kann_mopt_init(kann_mopt_t *mo)
{
	memset(mo, 0, sizeof(kann_mopt_t));
	mo->mini_algo = KANN_MM_RMSPROP;
	mo->batch_algo = KANN_MB_CONST;
	mo->lr = 0.001f;
	mo->fv = 0.1f;
	mo->max_mbs = 64;
	mo->epoch_lazy = 10;
	mo->max_epoch = 25;
	mo->max_rnn_len = 1;
}

kann_min_t *kann_minimizer(const kann_mopt_t *o, int n)
{
	kann_min_t *m;
	m = kann_min_new(o->mini_algo, o->batch_algo, n);
	kann_min_set_lr(m, o->lr);
	return m;
}

static void kann_count_class_error(const kad_node_t *q, int *n_err, int *n_err_tot)
{
	int j, len, k;
	kad_node_t *p = q->child[0].p, *r = q->child[1].p;
	len = kad_len(p) / p->d[0];
	for (j = k = 0; j < p->d[0]; ++j, k += len) {
		float p_max = -1e30f, r_sum = 0.0f, r_min = 1.0f, r_max = 0.0f;
		int i, p_max_i = -1, r_max_i = -1;
		for (i = 0; i < len; ++i) {
			if (p_max < p->x[k+i]) p_max = p->x[k+i], p_max_i = i;
			if (r_max < r->x[k+i]) r_max = r->x[k+i], r_max_i = i;
			if (r_min > r->x[k+i]) r_min = r->x[k+i];
			r_sum += r->x[k+i];
		}
		if (r_sum == 1.0f && r_min >= 0.0f) {
			++*n_err_tot;
			if (p_max_i != r_max_i) ++*n_err;
		}
	}
}

static inline int kann_get_cost_node(kann_t *a)
{
	int i, i_cost = -1, n_cost = 0;
	for (i = 0; i < a->n; ++i)
		if ((a->v[i]->ext_flag & KANN_F_COST) && a->v[i]->n_d == 0)
			i_cost = i, ++n_cost;
	assert(n_cost == 1);
	return i_cost;
}

float kann_grad(kann_t *a)
{
	int i_cost;
	float cost;
	i_cost = kann_get_cost_node(a);
	cost = *kad_eval_at(a->n, a->v, i_cost);
	kad_grad(a->n, a->v, i_cost);
	return cost;
}

static float kann_fnn_process_mini(kann_t *a, kann_min_t *m, int bs, float **x, float **y, int *n_err, int *n_err_tot) // train or validate a minibatch
{
	int i, i_cost;
	float cost;
	i_cost = kann_get_cost_node(a);
	kann_set_batch_size(a, bs);
	kann_bind_by_flag(a, KANN_F_IN, x);
	kann_bind_by_flag(a, KANN_F_TRUTH, y);
	cost = *kad_eval_at(a->n, a->v, i_cost);
	if (a->v[i_cost]->op == 10) {
		for (i = 0; i < a->v[i_cost]->n_child; ++i)
			kann_count_class_error(a->v[i_cost]->child[i].p, n_err, n_err_tot);
	} else kann_count_class_error(a->v[i_cost], n_err, n_err_tot);
	if (m) {
//		kad_check_grad(a->n, a->v, i_cost);
		kad_grad(a->n, a->v, i_cost);
		kann_min_mini_update(m, a->g, a->x);
	}
	return cost;
}

static float kann_process_batch(kann_t *a, kann_min_t *min, kann_reader_f rdr, void *data, int max_len, int max_mbs, kann_t *fnn_max, float **x, float **y, float *class_err)
{
	int n_in, n_out, tot = 0, n_err_tot = 0, n_err = 0, action;
	float cost = 0.0f, *x1, *y1;

	n_in = kann_n_in(a);
	n_out = kann_n_out(a);
	if (!kann_is_rnn(a)) max_len = 1;
	x1 = (float*)calloc(max_len * n_in,  sizeof(float));
	y1 = (float*)calloc(max_len * n_out, sizeof(float));
	action = min? KANN_RDR_READ_TRAIN : KANN_RDR_READ_VALIDATE;
	for (;;) {
		int i, k, l, len = -1;
		kann_t *fnn;
		rdr(data, KANN_RDR_MINI_RESET, max_len, 0, 0);
		for (k = 0; k < max_mbs; ++k) {
			if ((l = rdr(data, action, max_len, x1, y1)) <= 0 || (k > 0 && l != len)) break;
			len = l;
			for (i = 0; i < len; ++i) {
				memcpy(&x[i][k*n_in],  &x1[i*n_in],  n_in  * sizeof(float));
				memcpy(&y[i][k*n_out], &y1[i*n_out], n_out * sizeof(float));
			}
		}
		if (k == 0) break;
		fnn = len == max_len && fnn_max? fnn_max : kann_is_rnn(a)? kann_rnn_unroll(a, len) : a;
		cost += kann_fnn_process_mini(fnn, min, k, x, y, &n_err, &n_err_tot) * k;
		tot += k;
		if (fnn && fnn != fnn_max && fnn != a) kann_delete_unrolled(fnn);
	}
	free(y1); free(x1);
	if (class_err) *class_err = n_err_tot > 0? (float)n_err / n_err_tot : -1.0f;
	return cost / tot;
}

void kann_train(const kann_mopt_t *mo, kann_t *a, kann_reader_f rdr, void *data)
{
	float **x, **y, min_cost, *bak;
	int i, j, n_in, n_out, n_var, n_const, max_rnn_len, min_j, n_adj, streak;
	kann_min_t *min;
	kann_t *fnn_max = 0;

	n_in = kann_n_in(a);
	n_out = kann_n_out(a);
	n_var = kad_n_var(a->n, a->v);
	n_const = kad_n_const(a->n, a->v);
	max_rnn_len = kann_is_rnn(a)? mo->max_rnn_len : 1;
	if (max_rnn_len > 1) fnn_max = kann_rnn_unroll(a, max_rnn_len);

	x = (float**)malloc(max_rnn_len * sizeof(float*));
	y = (float**)malloc(max_rnn_len * sizeof(float*));
	for (i = 0; i < max_rnn_len; ++i) {
		x[i] = (float*)calloc(mo->max_mbs * n_in,  sizeof(float));
		y[i] = (float*)calloc(mo->max_mbs * n_out, sizeof(float));
	}

	bak = (float*)calloc(n_var * 2 + n_const, sizeof(float));
	min = kann_minimizer(mo, n_var);
	for (j = n_adj = streak = 0, min_cost = 1e30, min_j = -1; j < mo->max_epoch; ++j) {
		float running_cost = 0.0f, validate_cost = 0.0f, class_err = 0.0f;
		rdr(data, KANN_RDR_BATCH_RESET, 0, 0, 0);
		running_cost =  kann_process_batch(a, min, rdr, data, max_rnn_len, mo->max_mbs, 0, x, y, 0);
		validate_cost = kann_process_batch(a,   0, rdr, data, max_rnn_len, mo->max_mbs, 0, x, y, &class_err);
		kann_min_batch_finish(min, a->x);
		if (kann_verbose >= 3) {
			fprintf(stderr, "epoch: %d; running cost: %g; validation cost: %g", j+1, running_cost, validate_cost);
			if (class_err >= 0.0f) fprintf(stderr, " (class error: %.2f%%)", class_err * 100.0f);
			fputc('\n', stderr);
		}
		if (j >= mo->epoch_lazy) {
			if (validate_cost < min_cost) {
				streak = 0;
				min_j = j, min_cost = validate_cost;
				memcpy(bak, a->x, n_var * sizeof(float));
				memcpy(bak + n_var, a->g, n_var * sizeof(float));
				memcpy(bak + n_var * 2, a->c, n_const * sizeof(float));
			} else if (++streak >= mo->epoch_lazy) {
				streak = 0;
				memcpy(a->x, bak, n_var * sizeof(float));
				memcpy(a->g, bak + n_var, n_var * sizeof(float));
				memcpy(a->c, bak + n_var * 2, n_const * sizeof(float));
				if (++n_adj >= 2) break;
				kann_min_set_lr(min, .1f * mo->lr);
			}
		}
	}
	kann_min_delete(min);
	if (kann_verbose >= 3)
		fprintf(stderr, "min epoch: %d\n", min_j + 1);
	free(bak);

	for (i = 0; i < mo->max_rnn_len; ++i) {
		free(y[i]); free(x[i]);
	}
	free(y); free(x);
	if (fnn_max) kann_delete_unrolled(fnn_max);
}

/*****************************************
 *** @@APPLY: applying a trained model ***
 *****************************************/

const float *kann_apply1(kann_t *a, float *x)
{
	int i;
	kann_set_batch_size(a, 1);
	kann_bind_by_flag(a, KANN_F_IN, &x);
	kad_eval_flag(a->n, a->v, KANN_F_OUT);
	for (i = 0; i < a->n; ++i)
		if (a->v[i]->ext_flag & KANN_F_OUT)
			return a->v[i]->x;
	return 0;
}

void kann_rnn_start(kann_t *a)
{
	int i;
	kann_set_batch_size(a, 1);
	for (i = 0; i < a->n; ++i) {
		kad_node_t *p = a->v[i];
		if (p->pre) {
			kad_node_t *q = p->pre;
			if (q->x) memcpy(p->x, q->x, kad_len(p) * sizeof(float));
			else memset(p->x, 0, kad_len(p) * sizeof(float));
			q->x = p->x;
		}
	}
}

void kann_rnn_end(kann_t *a)
{
	kad_ext_sync(a->n, a->v, a->x, a->g, a->c);
}

float *kann_rnn_apply_seq1(kann_t *a, int len, float *x)
{
	float *y;
	int n_in, n_out, i;
	n_in = kann_n_in(a);
	n_out = kann_n_out(a);
	y = (float*)malloc(len * n_out * sizeof(float));
	kann_rnn_start(a);
	for (i = 0; i < len; ++i)
		memcpy(&y[i*n_out], kann_apply1(a, &x[i*n_in]), n_out * sizeof(float));
	kann_rnn_end(a);
	return y;
}

/*************************************************
 *** @@TRAIN_FNN: simpler API for training FNN ***
 *************************************************/

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

/*****************************
 *** @@MODEL_IO: model I/O ***
 *****************************/

#define KANN_MAGIC "KAN\1"

void kann_save_fp(FILE *fp, kann_t *ann)
{
	kann_set_batch_size(ann, 1);
	fwrite(KANN_MAGIC, 1, 4, fp);
	kad_save(fp, ann->n, ann->v);
	fwrite(ann->x, sizeof(float), kad_n_var(ann->n, ann->v), fp);
	fwrite(ann->c, sizeof(float), kad_n_const(ann->n, ann->v), fp);
}

void kann_save(const char *fn, kann_t *ann)
{
	FILE *fp;
	fp = fn && strcmp(fn, "-")? fopen(fn, "wb") : stdout;
	kann_save_fp(fp, ann);
	fclose(fp);
}

kann_t *kann_load_fp(FILE *fp)
{
	char magic[4];
	kann_t *ann;
	int n_var, n_const;

	fread(magic, 1, 4, fp);
	if (strncmp(magic, KANN_MAGIC, 4) != 0) {
		fclose(fp);
		return 0;
	}
	ann = (kann_t*)calloc(1, sizeof(kann_t));
	ann->v = kad_load(fp, &ann->n);
	n_var = kad_n_var(ann->n, ann->v);
	n_const = kad_n_const(ann->n, ann->v);
	ann->x = (float*)malloc(n_var * sizeof(float));
	ann->g = (float*)calloc(n_var, sizeof(float));
	ann->c = (float*)malloc(n_const * sizeof(float));
	fread(ann->x, sizeof(float), n_var, fp);
	fread(ann->c, sizeof(float), n_const, fp);
	kad_ext_sync(ann->n, ann->v, ann->x, ann->g, ann->c);
	return ann;
}

kann_t *kann_load(const char *fn)
{
	FILE *fp;
	kann_t *ann;
	fp = fn && strcmp(fn, "-")? fopen(fn, "rb") : stdin;
	ann = kann_load_fp(fp);
	fclose(fp);
	return ann;
}
