#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>
#include "kann_rand.h"
#include "kann_min.h"
#include "kann.h"

#define KANN_MAGIC "KAN\1"

int kann_verbose = 3;

kann_t *kann_new(void)
{
	return (kann_t*)calloc(1, sizeof(kann_t));
}

void kann_delete(kann_t *a)
{
	free(a->t); free(a->g);
	if (a->v) kad_delete(a->n, a->v);
	free(a);
}

kann_min_t *kann_minimizer(const kann_mopt_t *o, int n)
{
	kann_min_t *m;
	m = kann_min_new(KANN_MM_RMSPROP, KANN_MB_CONST, n);
	m->lr = o->lr, m->decay = o->decay;
	return m;
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
		if (kad_is_var(v)) {
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

int kann_is_rnn(const kann_t *a)
{
	int i;
	for (i = 0; i < a->n; ++i)
		if (a->v[i]->pre) return 1;
	return 0;
}

void kann_mopt_init(kann_mopt_t *mo)
{
	memset(mo, 0, sizeof(kann_mopt_t));
	mo->lr = 0.01f;
	mo->fv = 0.1f;
	mo->max_mbs = 64;
	mo->epoch_lazy = 10;
	mo->max_epoch = 100;
	mo->decay = 0.9f;
	mo->max_rnn_len = 1;
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

void kann_fnn_train(const kann_mopt_t *mo, kann_t *a, int n, float **x, float **y)
{
	void *data;
	data = kann_rdr_xy_new(n, mo->fv, kann_n_in(a), x, kann_n_out(a), y);
	kann_train(mo, a, kann_rdr_xy_read, data);
	kann_rdr_xy_delete(data);
}

const float *kann_fnn_apply1(kann_t *a, float *x) // FIXME: for now it doesn't work RNN
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

float *kann_rnn_apply_seq1(kann_t *a, int len, float *x)
{
	kann_t *fnn;
	float *y;
	int i, k, n_in, n_out;

	n_in = kann_n_in(a);
	n_out = kann_n_out(a);
	y = (float*)calloc(len * n_out, sizeof(float));
	fnn = kann_rnn_unroll(a, len, 0);
	kann_set_batch_size(fnn, 1);
	for (i = k = 0; i < a->n; ++i)
		if (a->v[i]->n_child == 0 && !a->v[i]->to_back && a->v[i]->label == KANN_L_IN)
			a->v[i]->x = &x[k], k += n_in;
	kad_eval_by_label(fnn->n, fnn->v, KANN_L_OUT);
	for (i = k = 0; i < fnn->n; ++i)
		if (fnn->v[i]->label == KANN_L_OUT) {
			memcpy(&y[k], fnn->v[i]->x, n_out * sizeof(float));
			k += n_out;
		}
	fnn->t = fnn->g = 0;
	kann_delete(fnn);
	return y;
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
	ann->v = kad_read(fp, &ann->n);
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
