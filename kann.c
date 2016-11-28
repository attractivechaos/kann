#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "kann.h"

#define KANN_MAGIC "KAN\1"

int kann_verbose = 3;

/***********************************
 *** Layers and model generation ***
 ***********************************/

kad_node_t *kann_new_weight(int n_row, int n_col)
{
	kad_node_t *w;
	w = kad_var(0, 0, 2, n_row, n_col);
	w->x = (float*)malloc(n_row * n_col * sizeof(float));
	kann_rand_weight(n_row, n_col, w->x);
	return w;
}

kad_node_t *kann_new_bias(int n)
{
	kad_node_t *b;
	b = kad_var(0, 0, 1, n);
	b->x = (float*)calloc(n, sizeof(float));
	return b;
}

kad_node_t *kann_layer_input(int n1)
{
	kad_node_t *t;
	t = kad_par(0, 2, 1, n1), t->label = KANN_L_IN;
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
	s = kad_par(0, 0), s->label = KANN_H_DROPOUT;
	s->x = (float*)calloc(1, sizeof(float));
	*s->x = r;
	return kad_dropout(t, s);
}

kad_node_t *kann_layer_rnn(kad_node_t *in, int n1, kann_activate_f af)
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
	s = kad_tanh(kad_add(kad_add(kad_cmul(in, w), kad_cmul(kad_mul(r, h0), u)), b));
	// h_t = z # h_{t-1} + (1 - z) # s
	out = kad_add(kad_mul(kad_1minus(z), s), kad_mul(z, h0));
	out->pre = h0;
	return out;
}

void kann_collate_x(kann_t *a)
{
	int i, j, k, l, n_par;
	n_par = kann_n_par(a);
	a->t = (float*)realloc(a->t, n_par * sizeof(float));
	a->g = (float*)realloc(a->g, n_par * sizeof(float));
	a->c = (float*)realloc(a->c, kann_n_hyper(a) * sizeof(float));
	memset(a->g, 0, n_par * sizeof(float));
	for (i = j = k = 0; i < a->n; ++i) {
		kad_node_t *v = a->v[i];
		if (kad_is_var(v)) {
			l = kad_len(v);
			memcpy(&a->t[j], v->x, l * sizeof(float));
			free(v->x);
			v->x = &a->t[j];
			v->g = &a->g[j];
			j += l;
		} else if (kann_is_hyper(v)) {
			l = kad_len(v);
			memcpy(&a->c[k], v->x, l * sizeof(float));
			free(v->x);
			v->x = &a->c[k];
			k += l;
		}
	}
}

void kann_sync_x(kann_t *a)
{
	int i, j, k;
	for (i = j = k = 0; i < a->n; ++i) {
		kad_node_t *v = a->v[i];
		if (kad_is_var(v)) {
			v->x = &a->t[j];
			v->g = &a->g[j];
			j += kad_len(v);
		} else if (kann_is_hyper(v)) {
			v->x = &a->c[k];
			k += kad_len(v);
		}
	}
}

kann_t *kann_layer_final(kad_node_t *t, int n_out, int type)
{
	kann_t *a = 0;
	kad_node_t *cost = 0, *truth = 0;
	assert(type == KANN_C_BIN_CE || type == KANN_C_CE);
	truth = kad_par(0, 2, 1, n_out), truth->label = KANN_L_TRUTH;
	t = kann_layer_linear(t, n_out);
	if (type == KANN_C_BIN_CE) {
		cost = kad_ce2(t, truth);
		t = kad_sigm(t);
	} else if (type == KANN_C_CE) {
		kad_node_t *temp;
		temp = kad_par(0, 0), temp->label = KANN_H_TEMP;
		temp->x = (float*)calloc(1, sizeof(float));
		*temp->x = 1.0f;
		cost = kad_cesm(t, truth);
		t = kad_softmax2(t, temp);
	}
	t->label = KANN_L_OUT, cost->label = KANN_L_COST;
	a = (kann_t*)calloc(1, sizeof(kann_t));
	a->v = kad_compile(&a->n, 2, t, cost);
	kann_collate_x(a);
	kad_drand = kann_drand;
	return a;
}

/**************************
 * Miscellaneous routines *
 **************************/

void kann_delete(kann_t *a)
{
	free(a->t); free(a->g); free(a->c);
	if (a->v) kad_delete(a->n, a->v);
	free(a);
}

void kann_set_hyper(kann_t *a, int label, float z)
{
	int i;
	for (i = 0; i < a->n; ++i)
		if (a->v[i]->label == label && a->v[i]->n_d == 0)
			*a->v[i]->x = z;
}

static void kann_set_batch_size(kann_t *a, int B)
{
	int i;
	for (i = 0; i < a->n; ++i)
		if (a->v[i]->label == KANN_L_IN || a->v[i]->label == KANN_L_TRUTH)
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

static int kann_bind_by_label(kann_t *a, int label, float **x)
{
	int i, k;
	for (i = k = 0; i < a->n; ++i)
		if (a->v[i]->n_child == 0 && !a->v[i]->to_back && a->v[i]->label == label)
			a->v[i]->x = x[k++];
	return k;
}

kann_t *kann_rnn_unroll(kann_t *a, int len, int pool_hidden)
{
	kann_t *b;
	b = (kann_t*)calloc(1, sizeof(kann_t));
	b->t = a->t, b->g = a->g;
	if (pool_hidden) {
		abort();
	} else {
		int i, n_root = 0, k;
		kad_node_t **t, **root;
		b->v = kad_unroll(a->n, a->v, len, &b->n);
		t = (kad_node_t**)calloc(len, sizeof(kad_node_t*));
		root = (kad_node_t**)calloc(len + 1, sizeof(kad_node_t*));
		for (i = k = 0; i < b->n; ++i) {
			if (b->v[i]->label == KANN_L_OUT) root[n_root++] = b->v[i];
			else if (b->v[i]->label == KANN_L_COST) t[k++] = b->v[i], b->v[i]->label = 0;
		}
		assert(k == len && n_root == len);
		root[n_root++] = kad_avg(k, t);
		root[n_root-1]->label = KANN_L_COST;
		free(b->v);
		b->v = kad_compile_array(&b->n, n_root, root);
		free(root); free(t);
	}
	return b;
}

/**********
 * Counts *
 **********/

static inline int kann_n_by_label(const kann_t *a, int label)
{
	int i, n = 0;
	for (i = 0; i < a->n; ++i)
		if (a->v[i]->label == label)
			n += a->v[i]->n_d > 1? kad_len(a->v[i]) / a->v[i]->d[0] : 1; // the first dimension is batch size
	return n;
}

int kann_n_in(const kann_t *a) { return kann_n_by_label(a, KANN_L_IN); }
int kann_n_out(const kann_t *a) { return kann_n_by_label(a, KANN_L_OUT); }

int kann_n_hyper(const kann_t *a)
{
	int i, n = 0;
	for (i = 0; i < a->n; ++i)
		if (kann_is_hyper(a->v[i]))
			n += kad_len(a->v[i]);
	return n;
}

int kann_is_rnn(const kann_t *a)
{
	int i;
	for (i = 0; i < a->n; ++i)
		if (a->v[i]->pre) return 1;
	return 0;
}

/******************
 * Model training *
 ******************/

void kann_mopt_init(kann_mopt_t *mo)
{
	memset(mo, 0, sizeof(kann_mopt_t));
	mo->lr = 0.001f;
	mo->fv = 0.1f;
	mo->max_mbs = 64;
	mo->epoch_lazy = 10;
	mo->max_epoch = 20;
	mo->decay = 0.9f;
	mo->max_rnn_len = 1;
}

kann_min_t *kann_minimizer(const kann_mopt_t *o, int n)
{
	kann_min_t *m;
	m = kann_min_new(KANN_MM_RMSPROP, KANN_MB_CONST, n);
	m->lr = o->lr, m->decay = o->decay;
	return m;
}

static float kann_fnn_process_mini(kann_t *a, kann_min_t *m, int bs, float **x, float **y) // train or validate a minibatch
{
	int i, i_cost = -1, n_cost = 0;
	float cost;
	for (i = 0; i < a->n; ++i)
		if (a->v[i]->label == KANN_L_COST)
			i_cost = i, ++n_cost;
	assert(n_cost == 1);
	kann_set_batch_size(a, bs);
	kann_bind_by_label(a, KANN_L_IN, x);
	kann_bind_by_label(a, KANN_L_TRUTH, y);
	cost = *kad_eval_from(a->n, a->v, i_cost);
	if (m) {
//		kad_check_grad(a->n, a->v, i_cost);
		kad_grad(a->n, a->v, i_cost);
		kann_min_mini_update(m, a->g, a->t);
	}
	return cost;
}

static float kann_process_batch(kann_t *a, kann_min_t *min, kann_reader_f rdr, void *data, int max_len, int max_mbs, kann_t *fnn_max, float **x, float **y)
{
	int n_in, n_out, tot = 0, action;
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
		fnn = len == max_len && fnn_max? fnn_max : kann_rnn_unroll(a, len, 0);
		cost += kann_fnn_process_mini(fnn, min, k, x, y) * k;
		tot += k;
		if (fnn && fnn != fnn_max) {
			fnn->t = fnn->g = 0;
			kann_delete(fnn);
		}
	}
	free(y1); free(x1);
	cost /= tot;
	return cost;
}

void kann_train(const kann_mopt_t *mo, kann_t *a, kann_reader_f rdr, void *data)
{
	float **x, **y;
	int i, j, n_in, n_out, n_par, max_rnn_len;
	kann_min_t *min;
	kann_t *fnn_max = 0;

	n_in = kann_n_in(a);
	n_out = kann_n_out(a);
	n_par = kann_n_par(a);
	max_rnn_len = kann_is_rnn(a)? mo->max_rnn_len : 1;
	if (max_rnn_len > 1) fnn_max = kann_rnn_unroll(a, max_rnn_len, 0);

	x = (float**)malloc(max_rnn_len * sizeof(float*));
	y = (float**)malloc(max_rnn_len * sizeof(float*));
	for (i = 0; i < max_rnn_len; ++i) {
		x[i] = (float*)calloc(mo->max_mbs * n_in,  sizeof(float));
		y[i] = (float*)calloc(mo->max_mbs * n_out, sizeof(float));
	}

	min = kann_minimizer(mo, n_par);
	for (j = 0; j < mo->max_epoch; ++j) {
		float running_cost = 0.0f, validate_cost = 0.0f;
		rdr(data, KANN_RDR_BATCH_RESET, 0, 0, 0);
		running_cost =  kann_process_batch(a, min, rdr, data, max_rnn_len, mo->max_mbs, 0, x, y);
		validate_cost = kann_process_batch(a,   0, rdr, data, max_rnn_len, mo->max_mbs, 0, x, y);
		kann_min_batch_finish(min, a->t);
		if (kann_verbose >= 3)
			fprintf(stderr, "epoch: %d; running cost: %g; validation cost: %g\n", j+1, running_cost, validate_cost);
	}
	kann_min_delete(min);

	for (i = 0; i < mo->max_rnn_len; ++i) {
		free(y[i]); free(x[i]);
	}
	free(y); free(x);
	if (fnn_max) {
		fnn_max->t = fnn_max->g = 0;
		kann_delete(fnn_max);
	}
}

/****************************
 * Applying a trained model *
 ****************************/

const float *kann_apply1(kann_t *a, float *x)
{
	int i;
	kann_set_batch_size(a, 1);
	kann_bind_by_label(a, KANN_L_IN, &x);
	kad_eval_by_label(a->n, a->v, KANN_L_OUT);
	for (i = 0; i < a->n; ++i)
		if (a->v[i]->label == KANN_L_OUT)
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
			memcpy(p->x, q->x, kad_len(p) * sizeof(float));
			q->ptr = q->x;
			q->x = p->x;
		}
	}
}

void kann_rnn_end(kann_t *a)
{
	int i;
	for (i = 0; i < a->n; ++i) {
		kad_node_t *p = a->v[i];
		if (p->pre) p->pre->x = (float*)p->pre->ptr;
	}
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

/*************
 * Model I/O *
 *************/

void kann_write(const char *fn, const kann_t *ann)
{
	FILE *fp;
	fp = fn && strcmp(fn, "-")? fopen(fn, "wb") : stdout;
	kann_write_core(fp, ann);
	fclose(fp);
}

void kann_write_core(FILE *fp, const kann_t *ann)
{
	fwrite(KANN_MAGIC, 1, 4, fp);
	kad_write(fp, ann->n, ann->v);
	fwrite(ann->t, sizeof(float), kann_n_par(ann), fp);
	fwrite(ann->c, sizeof(float), kann_n_hyper(ann), fp);
}

kann_t *kann_read(const char *fn)
{
	FILE *fp;
	kann_t *ann;
	fp = fn && strcmp(fn, "-")? fopen(fn, "rb") : stdin;
	ann = kann_read_core(fp);
	fclose(fp);
	return ann;
}

kann_t *kann_read_core(FILE *fp)
{
	char magic[4];
	kann_t *ann;
	int n_par, n_hyper;

	fread(magic, 1, 4, fp);
	if (strncmp(magic, KANN_MAGIC, 4) != 0) {
		fclose(fp);
		return 0;
	}
	ann = (kann_t*)calloc(1, sizeof(kann_t));
	ann->v = kad_read(fp, &ann->n);
	n_par = kann_n_par(ann);
	n_hyper = kann_n_hyper(ann);
	ann->t = (float*)malloc(n_par * sizeof(float));
	ann->g = (float*)calloc(n_par, sizeof(float));
	ann->c = (float*)malloc(n_hyper * sizeof(float));
	fread(ann->t, sizeof(float), n_par, fp);
	fread(ann->c, sizeof(float), n_hyper, fp);
	kann_sync_x(ann);
	return ann;
}

/**************************************
 *** Pseudo-random number generator ***
 **************************************/

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

/*****************
 *** Minimizer ***
 *****************/

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
	kann_min_t *m;
	if (mini_algo <= 0) mini_algo = KANN_MM_RMSPROP;
	if (batch_algo <= 0) batch_algo = KANN_MB_CONST;
	m = (kann_min_t*)calloc(1, sizeof(kann_min_t));
	m->mini_algo = mini_algo, m->batch_algo = batch_algo, m->n = n;
	if (mini_algo == KANN_MM_RMSPROP) {
		m->lr = 0.001f, m->decay = 0.9f;
		m->maux = (float*)calloc(n, sizeof(float));
	}
	return m;
}

void kann_min_delete(kann_min_t *m)
{
	free(m->maux); free(m->baux); free(m);
}

void kann_min_mini_update(kann_min_t *m, const float *g, float *t)
{
	if (m->mini_algo == KANN_MM_RMSPROP)
		kann_RMSprop(m->n, m->lr, 0, m->decay, g, t, m->maux);
}

void kann_min_batch_finish(kann_min_t *m, const float *t)
{
}
