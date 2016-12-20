#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>
#include <float.h>
#include <math.h>
#include "kautodiff.h"

kad_drand_f kad_drand = drand48;

/**********************
 * Graph construction *
 **********************/

static inline kad_node_t *kad_new_core(int n_d, int op, int n_child)
{
	kad_node_t *s;
	if (n_d > KAD_MAX_DIM) return 0;
	s = (kad_node_t*)calloc(1, sizeof(kad_node_t));
	s->n_d = n_d, s->op = op, s->n_child = n_child;
	if (s->n_child) s->child = (kad_edge_t*)calloc(s->n_child, sizeof(kad_edge_t));
	return s;
}

static inline kad_node_t *kad_new_external(float *x, float *g, int n_d, va_list ap)
{
	kad_node_t *p;
	int i;
	p = kad_new_core(n_d, 0, 0);
	for (i = 0; i < n_d; ++i)
		p->d[i] = va_arg(ap, int);
	p->x = x, p->g = g;
	return p;
}

kad_node_t *kad_const(float *x, int n_d, ...)
{
	kad_node_t *p;
	va_list ap;
	va_start(ap, n_d);
	p = kad_new_external(x, 0, n_d, ap);
	va_end(ap);
	p->flag |= KAD_F_CONSTANT;
	return p;
}

kad_node_t *kad_feed(float *x, int n_d, ...)
{
	kad_node_t *p;
	va_list ap;
	va_start(ap, n_d);
	p = kad_new_external(x, 0, n_d, ap);
	va_end(ap);
	return p;
}

kad_node_t *kad_var(float *x, float *g, int n_d, ...)
{
	kad_node_t *p;
	va_list ap;
	va_start(ap, n_d);
	p = kad_new_external(x, 0, n_d, ap);
	va_end(ap);
	p->flag |= KAD_F_WITH_PD;
	return p;
}

static inline kad_node_t *kad_op2_core(int op, kad_node_t *x, kad_node_t *y)
{
	kad_node_t *s;
	s = kad_new_core(0, op, 2);
	s->child[0].p = x, s->child[1].p = y;
	if (kad_op_list[op](s, KAD_SYNC_DIM) < 0) {
		free(s->child); free(s);
		return 0;
	}
	return s;
}

static inline kad_node_t *kad_op1_core(int op, kad_node_t *x)
{
	kad_node_t *s;
	s = kad_new_core(0, op, 1);
	s->child[0].p = x;
	kad_op_list[op](s, KAD_SYNC_DIM);
	return s;
}

#define KAD_FUNC_OP2(fname, op) kad_node_t *fname(kad_node_t *x, kad_node_t *y) { return kad_op2_core((op), x, y); }

KAD_FUNC_OP2(kad_add, 1)
KAD_FUNC_OP2(kad_mul, 2)
KAD_FUNC_OP2(kad_cmul, 3)
KAD_FUNC_OP2(kad_ceb, 4)
KAD_FUNC_OP2(kad_matmul, 9)
KAD_FUNC_OP2(kad_cem, 12)
KAD_FUNC_OP2(kad_softmax2, 13)
KAD_FUNC_OP2(kad_dropout, 15)

#define KAD_FUNC_OP1(fname, op) kad_node_t *fname(kad_node_t *x) { return kad_op1_core((op), x); }

KAD_FUNC_OP1(kad_norm2, 5)
KAD_FUNC_OP1(kad_sigm, 6)
KAD_FUNC_OP1(kad_tanh, 7)
KAD_FUNC_OP1(kad_relu, 8)
KAD_FUNC_OP1(kad_1minus, 11)
KAD_FUNC_OP1(kad_softmax, 14)

/////////// General pooling ///////////

static kad_node_t *kad_op_pooling_core(int op, int n, kad_node_t **x)
{
	int i;
	kad_node_t *s;
	s = kad_new_core(0, op, n);
	s->flag |= KAD_F_POOLING;
	for (i = 0; i < n; ++i)
		s->child[i].p = x[i];
	if (kad_op_list[op](s, KAD_SYNC_DIM) < 0) {
		free(s->child); free(s);
		return 0;
	}
	return s;
}

kad_node_t *kad_avg(int n, kad_node_t **x) { return kad_op_pooling_core(10, n, x); }
kad_node_t *kad_max(int n, kad_node_t **x) { return kad_op_pooling_core(21, n, x); }

/////////// Convolution ///////////

// compute output dimension and padding sizes on both sides
static inline int conv_find_par(int in_size, int kernel_size, int stride, int pad0, int *new_pad0, int *new_pad1)
{
	int out_size, pad_both;
	// key equation: out_size = (in_size - kernel_size + pad_both) / stride + 1
	if (pad0 == KAD_PAD_SAME && stride == 1) out_size = in_size;
	else out_size = (in_size - kernel_size + (pad0 > 0? pad0 : 0) + stride - 1) / stride + 1;
	pad_both = (out_size - 1) * stride + kernel_size - in_size;
	*new_pad0 = pad_both / 2;
	*new_pad1 = pad_both - *new_pad0;
	return out_size;
}

typedef struct {
	int kernel_size, stride, pad[2];
} conv_conf_t;

static inline conv_conf_t *conv2d_gen_aux(int in_row, int in_col, int kernel_r, int kernel_c, int stride_r, int stride_c, int top_pad, int left_pad)
{
	conv_conf_t *cnn;
	cnn = (conv_conf_t*)calloc(2, sizeof(conv_conf_t));
	cnn[0].kernel_size = kernel_r, cnn[0].stride = stride_r;
	cnn[1].kernel_size = kernel_c, cnn[1].stride = stride_c;
	conv_find_par(in_row, kernel_r, stride_r, top_pad,  &cnn[0].pad[0], &cnn[0].pad[1]);
	conv_find_par(in_col, kernel_c, stride_c, left_pad, &cnn[1].pad[0], &cnn[1].pad[1]);
	return cnn;
}

kad_node_t *kad_conv2d(kad_node_t *x, kad_node_t *w, int stride_r, int stride_c, int top_pad, int left_pad)
{
	kad_node_t *s;
	if (x->n_d != 4 || w->n_d != 4) return 0;
	s = kad_new_core(0, 16, 2);
	s->child[0].p = x, s->child[1].p = w;
	s->ptr = conv2d_gen_aux(x->d[2], x->d[3], w->d[2], w->d[3], stride_r, stride_c, top_pad, left_pad);
	s->ptr_size = sizeof(conv_conf_t) * 2;
	if (kad_op_list[16](s, KAD_SYNC_DIM) < 0) {
		free(s->ptr); free(s->child); free(s);
		return 0;
	}
	return s;
}

kad_node_t *kad_max2d(kad_node_t *x, int kernel_r, int kernel_c, int stride_r, int stride_c, int top_pad, int left_pad)
{
	kad_node_t *s;
	if (x->n_d != 4) return 0;
	s = kad_new_core(0, 17, 1);
	s->child[0].p = x;
	s->ptr = conv2d_gen_aux(x->d[2], x->d[3], kernel_r, kernel_c, stride_r, stride_c, top_pad, left_pad);
	s->ptr_size = sizeof(conv_conf_t) * 2;
	if (kad_op_list[17](s, KAD_SYNC_DIM) < 0) {
		free(s->ptr); free(s->child); free(s);
		return 0;
	}
	return s;
}

static inline conv_conf_t *conv1d_gen_aux(int in_col, int kernel_c, int stride_c, int left_pad)
{
	conv_conf_t *cnn;
	cnn = (conv_conf_t*)calloc(1, sizeof(conv_conf_t));
	cnn->kernel_size = kernel_c, cnn->stride = stride_c;
	conv_find_par(in_col, kernel_c, stride_c, left_pad, &cnn->pad[0], &cnn->pad[1]);
	return cnn;
}

kad_node_t *kad_conv1d(kad_node_t *x, kad_node_t *w, int stride, int left_pad)
{
	kad_node_t *s;
	if (x->n_d != 3 || w->n_d != 3) return 0;
	s = kad_new_core(0, 18, 2);
	s->child[0].p = x, s->child[1].p = w;
	s->ptr = conv1d_gen_aux(x->d[2], w->d[2], stride, left_pad);
	s->ptr_size = sizeof(conv_conf_t);
	if (kad_op_list[18](s, KAD_SYNC_DIM) < 0) {
		free(s->ptr); free(s->child); free(s);
		return 0;
	}
	return s;
}

kad_node_t *kad_max1d(kad_node_t *x, int kernel_size, int stride, int left_pad)
{
	kad_node_t *s;
	if (x->n_d != 3) return 0;
	s = kad_new_core(0, 19, 1);
	s->child[0].p = x;
	s->ptr = conv1d_gen_aux(x->d[2], kernel_size, stride, left_pad);
	s->ptr_size = sizeof(conv_conf_t);
	if (kad_op_list[19](s, KAD_SYNC_DIM) < 0) {
		free(s->ptr); free(s->child); free(s);
		return 0;
	}
	return s;
}

/////////// Miscellaneous ///////////

kad_node_t *kad_split(kad_node_t *x, int dim, int start, int end)
{
	kad_node_t *s;
	int32_t *aux;
	if (end < start || start < 0) return 0;
	aux = (int32_t*)malloc(3 * 4);
	aux[0] = dim, aux[1] = start, aux[2] = end;
	s = kad_new_core(0, 20, 1);
	s->child[0].p = x;
	s->ptr = aux, s->ptr_size = 3 * 4;
	if (kad_op_list[20](s, KAD_SYNC_DIM) < 0) {
		free(aux); free(s->child); free(s);
		return 0;
	}
	return s;
}

/***********************
 * Graph linearization *
 ***********************/

static void kad_mark_back(int n, kad_node_t **v)
{
	int i, j;
	for (i = 0; i < n; ++i)
		for (j = 0; j < v[i]->n_child; ++j)
			if (kad_is_back(v[i]->child[j].p))
				v[i]->flag |= KAD_F_WITH_PD;
}

static void kad_allocate_internal(int n, kad_node_t **v)
{
	int i;
	kad_mark_back(n, v);
	for (i = 0; i < n; ++i) {
		kad_node_t *p = v[i];
		if (p->n_child == 0) continue;
		p->x = (float*)realloc(p->x, kad_len(p) * sizeof(float));
		if (kad_is_back(p)) {
			p->g = (float*)realloc(p->g, kad_len(p) * sizeof(float));
			kad_op_list[p->op](p, KAD_ALLOC);
		}
	}
}

#define kvec_t(type) struct { size_t n, m; type *a; }

#define kv_pop(v) ((v).a[--(v).n])

#define kv_push(type, v, x) do { \
		if ((v).n == (v).m) { \
			(v).m = (v).m? (v).m<<1 : 2; \
			(v).a = (type*)realloc((v).a, sizeof(type) * (v).m); \
		} \
		(v).a[(v).n++] = (x); \
	} while (0)

typedef struct kad_node_t *kad_node_p;

// IMPORTANT: kad_node_t::tmp MUST BE set to zero before calling this function
kad_node_t **kad_compile_array(int *n_node, int n_roots, kad_node_t **roots)
{
	int i;
	kvec_t(kad_node_p) stack = {0,0,0}, a = {0,0,0};

	// generate kad_node_t::tmp
	for (i = 0; i < n_roots; ++i) kv_push(kad_node_p, stack, roots[i]);
	while (stack.n) {
		kad_node_t *p = kv_pop(stack);
		for (i = 0; i < p->n_child; ++i) {
			kad_node_t *q = p->child[i].p;
			if (q->tmp == 0) kv_push(kad_node_p, stack, q);
			++q->tmp;
		}
	}
	for (i = 0; i < n_roots; ++i) // check if roots are really roots
		assert(roots[i]->tmp == 0);

	// topological sorting (Kahn's algorithm)
	for (i = 0; i < n_roots; ++i) kv_push(kad_node_p, stack, roots[i]);
	while (stack.n) {
		kad_node_t *p = kv_pop(stack);
		kv_push(kad_node_p, a, p);
		for (i = 0; i < p->n_child; ++i)
			if (--p->child[i].p->tmp == 0)
				kv_push(kad_node_p, stack, p->child[i].p);
	}
	free(stack.a);
	for (i = 0; i < a.n; ++i) // check cycles; no cycles if constructed with kad_add() etc
		assert(a.a[i]->tmp == 0);

	// post-processing: reverse, mark back flag and allocate memory for internal nodes
	for (i = 0; i < a.n>>1; ++i) { // reverse a.a[]
		kad_node_p t;
		t = a.a[i], a.a[i] = a.a[a.n-1-i], a.a[a.n-1-i] = t;
	}
	kad_allocate_internal(a.n, a.a);

	*n_node = a.n;
	return a.a;
}

kad_node_t **kad_compile(int *n_node, int n_roots, ...)
{
	int i;
	kad_node_t **roots;
	va_list ap;

	roots = (kad_node_t**)alloca(n_roots * sizeof(kad_node_t*));
	va_start(ap, n_roots);
	for (i = 0; i < n_roots; ++i) roots[i] = va_arg(ap, kad_node_p);
	va_end(ap);
	return kad_compile_array(n_node, n_roots, roots);
}

/************************************
 * Miscellaneous on compiled graphs *
 ************************************/

void kad_delete(int n, kad_node_t **a)
{
	int i, j;
	for (i = 0; i < n; ++i) {
		kad_node_t *p = a[i];
		for (j = 0; j < p->n_child; ++j)
			free(p->child[j].t);
		if (p->n_child) {
			free(p->x);
			free(p->g);
		}
		free(p->child);
		free(p->ptr);
		free(p);
	}
	free(a);
}

int kad_n_var(int n, kad_node_t *const* v)
{
	int c, i;
	for (i = c = 0; i < n; ++i)
		if (kad_is_var(v[i]))
			c += kad_len(v[i]);
	return c;
}

int kad_n_const(int n, kad_node_t *const* v)
{
	int c, i;
	for (i = c = 0; i < n; ++i)
		if (kad_is_const(v[i]))
			c += kad_len(v[i]);
	return c;
}

void kad_ext_collate(int n, kad_node_t **a, float **_x, float **_g, float **_c)
{
	int i, j, k, l, n_var;
	float *x, *g, *c;
	n_var = kad_n_var(n, a);
	x = *_x = (float*)realloc(*_x, n_var * sizeof(float));
	g = *_g = (float*)realloc(*_g, n_var * sizeof(float));
	c = *_c = (float*)realloc(*_c, kad_n_const(n, a) * sizeof(float));
	memset(g, 0, n_var * sizeof(float));
	for (i = j = k = 0; i < n; ++i) {
		kad_node_t *v = a[i];
		if (kad_is_var(v)) {
			l = kad_len(v);
			memcpy(&x, v->x, l * sizeof(float));
			free(v->x);
			v->x = &x[j];
			v->g = &g[j];
			j += l;
		} else if (kad_is_const(v)) {
			l = kad_len(v);
			memcpy(&c[k], v->x, l * sizeof(float));
			free(v->x);
			v->x = &c[k];
			k += l;
		}
	}
}

void kad_ext_sync(int n, kad_node_t **a, float *x, float *g, float *c)
{
	int i, j, k;
	for (i = j = k = 0; i < n; ++i) {
		kad_node_t *v = a[i];
		if (kad_is_var(v)) {
			v->x = &x[j];
			v->g = &g[j];
			j += kad_len(v);
		} else if (kad_is_const(v)) {
			v->x = &c[k];
			k += kad_len(v);
		}
	}
}

/**********************************
 * Computate values and gradients *
 **********************************/

static void kad_mark_compute(int n, kad_node_t **a)
{
	int i, j;
	for (i = n - 1; i >= 0; --i)
		if (a[i]->tmp)
			for (j = 0; j < a[i]->n_child; ++j)
				a[i]->child[j].p->tmp = 1;
}

static void kad_eval_core(int n, kad_node_t **a)
{
	int i;
	kad_mark_compute(n, a);
	for (i = 0; i < n; ++i)
		if (a[i]->n_child && a[i]->tmp)
			kad_op_list[a[i]->op](a[i], KAD_FORWARD);
	for (i = 0; i < n; ++i) a[i]->tmp = 0;
}

const float *kad_eval_at(int n, kad_node_t **a, int from)
{
	int i;
	if (from < 0 || from >= n) from = n - 1;
	for (i = 0; i < n; ++i) a[i]->tmp = (i == from);
	kad_eval_core(n, a);
	return a[from]->x;
}

void kad_eval_flag(int n, kad_node_t **a, int ext_flag)
{
	int i;
	for (i = 0; i < n; ++i) a[i]->tmp = (a[i]->ext_flag & ext_flag)? 1 : 0;
	kad_eval_core(n, a);
}

void kad_grad(int n, kad_node_t **a, int from)
{
	int i;
	if (from < 0 || from >= n) from = n - 1;
	assert(a[from]->n_d == 0);
	for (i = 0; i < n; ++i) a[i]->tmp = (i == from);
	kad_mark_compute(n, a);
	for (i = 0; i <= from; ++i) // set all grandients to zero
		if (a[i]->g && a[i]->tmp) memset(a[i]->g, 0, kad_len(a[i]) * sizeof(float));
	for (i = from, a[i]->g[0] = 1.0f; i >= 0; --i) // backprop
		if (a[i]->n_child && a[i]->tmp)
			kad_op_list[a[i]->op](a[i], KAD_BACKWARD);
	for (i = 0; i <= from; ++i) a[i]->tmp = 0;
}

/***********************
 * Load and save graph *
 ***********************/

static void kad_write1(FILE *fp, const kad_node_t *p)
{
	fwrite(&p->ext_label, 4, 1, fp);
	fwrite(&p->ext_flag, 4, 1, fp);
	fwrite(&p->n_child, 4, 1, fp);
	if (p->n_child) {
		int32_t j, pre = p->pre? p->pre->tmp : -1;
		fwrite(&p->op, 2, 1, fp);
		for (j = 0; j < p->n_child; ++j)
			fwrite(&p->child[j].p->tmp, 4, 1, fp);
		fwrite(&pre, 4, 1, fp);
		fwrite(&p->ptr_size, 4, 1, fp);
		if (p->ptr_size > 0 && p->ptr)
			fwrite(p->ptr, p->ptr_size, 1, fp);
	} else {
		fwrite(&p->n_d, 1, 1, fp);
		if (p->n_d) fwrite(p->d, 4, p->n_d, fp);
		fwrite(&p->flag, 1, 1, fp);
	}
}

static kad_node_t *kad_read1(FILE *fp, kad_node_t **node)
{
	kad_node_t *p;
	p = (kad_node_t*)calloc(1, sizeof(kad_node_t));
	fread(&p->ext_label, 4, 1, fp);
	fread(&p->ext_flag, 4, 1, fp);
	fread(&p->n_child, 4, 1, fp);
	if (p->n_child) {
		int32_t j, k;
		p->child = (kad_edge_t*)calloc(p->n_child, sizeof(kad_edge_t));
		fread(&p->op, 2, 1, fp);
		for (j = 0; j < p->n_child; ++j) {
			fread(&k, 4, 1, fp);
			p->child[j].p = node? node[k] : 0;
		}
		fread(&k, 4, 1, fp);
		if (k >= 0) p->pre = node[k];
		fread(&p->ptr_size, 4, 1, fp);
		if (p->ptr_size > 0) {
			p->ptr = malloc(p->ptr_size);
			fread(p->ptr, p->ptr_size, 1, fp);
		}
	} else {
		fread(&p->n_d, 1, 1, fp);
		if (p->n_d) fread(p->d, 4, p->n_d, fp);
		fread(&p->flag, 1, 1, fp);
	}
	return p;
}

int kad_write(FILE *fp, int n_node, kad_node_t **node)
{
	int32_t i, k = n_node;
	fwrite(&k, 4, 1, fp);
	for (i = 0; i < n_node; ++i) node[i]->tmp = i;
	for (i = 0; i < n_node; ++i) kad_write1(fp, node[i]);
	for (i = 0; i < n_node; ++i) node[i]->tmp = 0;
	return 0;
}

kad_node_t **kad_read(FILE *fp, int *_n_node)
{
	int32_t i, n_node;
	kad_node_t **node;
	fread(&n_node, 4, 1, fp);
	node = (kad_node_t**)malloc(n_node * sizeof(kad_node_t*));
	for (i = 0; i < n_node; ++i) {
		kad_node_t *p;
		p = node[i] = kad_read1(fp, node);
		if (p->n_child) {
			kad_op_list[p->op](p, KAD_ALLOC);
			kad_op_list[p->op](p, KAD_SYNC_DIM);
		}
	}
	*_n_node = n_node;
	kad_mark_back(n_node, node);
	return node;
}

/**************
 * Unroll RNN *
 **************/

static inline kad_node_t *kad_dup1(const kad_node_t *p)
{
	kad_node_t *q;
	q = (kad_node_t*)malloc(sizeof(kad_node_t));
	memcpy(q, p, sizeof(kad_node_t));
	q->pre = 0, q->tmp = 0;
	if (q->n_child) {
		q->x = q->g = 0;
		q->child = (kad_edge_t*)calloc(q->n_child, sizeof(kad_edge_t));
	}
	return q;
}

kad_node_t **kad_unroll(int n_v, kad_node_t **v, int len, int *new_n)
{
	int i, j, k, l, k0;
	short *flag;
	kad_node_t **w, **alt;

	// set flags
	flag = (short*)calloc(n_v, sizeof(short));
	for (i = 0; i < n_v; ++i) {
		v[i]->tmp = i;
		if (kad_is_var(v[i]) || kad_is_const(v[i])) flag[i] |= 1; // external nodes that should not be duplicated
		if (v[i]->pre) flag[v[i]->pre->tmp] |= 2;
		if (kad_is_pivot(v[i])) {
			assert(v[i]->n_child == 1);
			flag[v[i]->child[0].p->tmp] |= 4; // parent is a pooling node
			flag[i] |= 8;
		}
		for (j = 0; j < v[i]->n_child; ++j)
			if (flag[v[i]->child[j].p->tmp]&8) flag[i] |= 8; // a node that can't be unrolled
	}

	// unroll unrollable nodes
	w = (kad_node_t**)calloc(n_v * len, sizeof(kad_node_t*));
	alt = (kad_node_t**)calloc(n_v, sizeof(kad_node_t*));
	for (l = k = 0; l < len; ++l) {
		for (i = 0; i < n_v; ++i) {
			if (flag[i]&8) continue;
			if (l > 0 && (flag[i]&3)) continue;
			w[k] = kad_dup1(v[i]);
			if (w[k]->n_child) {
				w[k]->x = w[k]->g = 0;
				for (j = 0; j < w[k]->n_child; ++j)
					w[k]->child[j].p = alt[v[i]->child[j].p->tmp];
			}
			w[k]->tmp = (flag[i]&4)? i : -1;
			if (v[i]->pre) alt[v[i]->pre->tmp] = w[k];
			alt[i] = w[k++];
		}
	}
	k0 = k;

	// unroll the rest of nodes
	for (i = 0; i < n_v; ++i) {
		if (!(flag[i]&8)) continue;
		assert(v[i]->pre == 0);
		w[k] = kad_dup1(v[i]);
		if (kad_is_pivot(v[i])) {
			w[k]->n_child = len, w[k]->tmp = 0;
			w[k]->child = (kad_edge_t*)realloc(w[k]->child, len * sizeof(kad_edge_t));
			memset(w[k]->child, 0, len * sizeof(kad_edge_t));
		} else if (w[k]->n_child) {
			w[k]->x = w[k]->g = 0;
			for (j = 0; j < w[k]->n_child; ++j)
				w[k]->child[j].p = alt[v[i]->child[j].p->tmp];
		}
		alt[i] = w[k++];
	}

	// pool
	for (i = 0; i < n_v; ++i)
		if (kad_is_pivot(v[i]))
			alt[v[i]->child[0].p->tmp] = alt[i];
	for (i = 0; i < k0; ++i) {
		if (w[i]->tmp >= 0) {
			kad_node_t *q = alt[w[i]->tmp];
			q->child[q->tmp++].p = w[i];
		}
		w[i]->tmp = 0;
	}
	for (i = 0; i < n_v; ++i) v[i]->tmp = 0;

	free(alt); free(flag);
	kad_allocate_internal(k, w);
	*new_n = k;
	return w;
}

/*********************
 * Vector operations *
 *********************/

#ifdef __SSE__
#include <xmmintrin.h>

static inline float kad_sdot(int n, const float *x, const float *y) // BLAS sdot using SSE
{
	int i, n8 = n>>3<<3;
	__m128 vs1, vs2;
	float s, t[4];
	vs1 = _mm_setzero_ps();
	vs2 = _mm_setzero_ps();
	for (i = 0; i < n8; i += 8) {
		__m128 vx1, vx2, vy1, vy2;
		vx1 = _mm_loadu_ps(&x[i]);
		vx2 = _mm_loadu_ps(&x[i+4]);
		vy1 = _mm_loadu_ps(&y[i]);
		vy2 = _mm_loadu_ps(&y[i+4]);
		vs1 = _mm_add_ps(vs1, _mm_mul_ps(vx1, vy1));
		vs2 = _mm_add_ps(vs2, _mm_mul_ps(vx2, vy2));
	}
	for (s = 0.; i < n; ++i) s += x[i] * y[i];
	_mm_storeu_ps(t, vs1);
	s += t[0] + t[1] + t[2] + t[3];
	_mm_storeu_ps(t, vs2);
	s += t[0] + t[1] + t[2] + t[3];
	return s;
}
static inline void kad_saxpy(int n, float a, const float *x, float *y) // BLAS saxpy using SSE
{
	int i, n8 = n>>3<<3;
	__m128 va;
	va = _mm_set1_ps(a);
	for (i = 0; i < n8; i += 8) {
		__m128 vx1, vx2, vy1, vy2, vt1, vt2;
		vx1 = _mm_loadu_ps(&x[i]);
		vx2 = _mm_loadu_ps(&x[i+4]);
		vy1 = _mm_loadu_ps(&y[i]);
		vy2 = _mm_loadu_ps(&y[i+4]);
		vt1 = _mm_add_ps(_mm_mul_ps(va, vx1), vy1);
		vt2 = _mm_add_ps(_mm_mul_ps(va, vx2), vy2);
		_mm_storeu_ps(&y[i], vt1);
		_mm_storeu_ps(&y[i+4], vt2);
	}
	for (; i < n; ++i) y[i] += a * x[i];
}
#else
static inline float kad_sdot(int n, const float *x, const float *y) // BLAS sdot
{
	int i;
	float s = 0.;
	for (i = 0; i < n; ++i) s += x[i] * y[i];
	return s;
}
static inline void kad_saxpy(int n, float a, const float *x, float *y) // BLAS saxpy
{
	int i;
	for (i = 0; i < n; ++i) y[i] += a * x[i];
}
#endif

void kad_vec_mul_sum(int n, float *a, const float *b, const float *c)
{
	int i;
	for (i = 0; i < n; ++i) a[i] += b[i] * c[i];
}

#ifdef HAVE_CBLAS
#include <cblas.h>
void kad_sgemm_simple(int trans_A, int trans_B, int M, int N, int K, const float *A, const float *B, float *C)
{
	cblas_sgemm(CblasRowMajor, trans_A? CblasTrans : CblasNoTrans, trans_B? CblasTrans : CblasNoTrans, M, N, K, 1.0f, A, trans_A? M : K, B, trans_B? K : N, 1.0f, C, N);
}
#else
void kad_sgemm_simple(int trans_A, int trans_B, int M, int N, int K, const float *A, const float *B, float *C) // simplified BLAS sgemm
{
	static const int x = 16;
	int i, j, k;
	if (!trans_A && trans_B) {
		for (i = 0; i < M; i += x)
			for (j = 0; j < N; j += x) {
				int ii, ie = M < i + x? M : i + x;
				int jj, je = N < j + x? N : j + x;
				for (ii = i; ii < ie; ++ii) { // loop tiling
					const float *aii = A + ii * K, *bjj;
					float *cii = C + ii * N;
					for (jj = j, bjj = B + j * K; jj < je; ++jj, bjj += K)
						cii[jj] += kad_sdot(K, aii, bjj);
				}
			}
	} else if (!trans_A && !trans_B) {
		for (i = 0; i < M; ++i)
			for (k = 0; k < K; ++k)
				kad_saxpy(N, A[i*K+k], &B[k*N], &C[i*N]);
	} else if (trans_A && !trans_B) {
		for (k = 0; k < K; ++k)
			for (i = 0; i < M; ++i)
				kad_saxpy(N, A[k*M+i], &B[k*N], &C[i*N]);
	} else abort(); // not implemented for (trans_A && trans_B)
}
#endif

/*************
 * Operators *
 *************/

static inline void kad_sync_dim1(kad_node_t *dst, const kad_node_t *src) // set the dimension/shape of dst to src
{
	dst->n_d = src->n_d;
	if (src->n_d) memcpy(dst->d, src->d, src->n_d * sizeof(int));
}

/////////// Arithmetic operations ///////////

int kad_op_add(kad_node_t *p, int action)
{
	int i, n0, n1;
	kad_node_t *q[2];

	q[0] = p->child[0].p, n0 = kad_len(q[0]);
	q[1] = p->child[1].p, n1 = kad_len(q[1]);
	if (action == KAD_SYNC_DIM) {
		if (n0 % n1 != 0) return -1;
		kad_sync_dim1(p, q[0]);
	} else if (action == KAD_FORWARD) {
		assert(n0 >= n1);
		memcpy(p->x, q[0]->x, n0 * sizeof(float));
		for (i = 0; i < n0; i += n1)
			kad_saxpy(n1, 1.0f, q[1]->x, p->x + i);
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(q[0])) kad_saxpy(n0, 1.0f, p->g, q[0]->g);
		if (kad_is_back(q[1]))
			for (i = 0; i < n0; i += n1)
				kad_saxpy(n1, 1.0f, p->g + i, q[1]->g);
	}
	return 0;
}

int kad_op_mul(kad_node_t *p, int action)
{
	int i, n0, n1;
	kad_node_t *q[2];

	q[0] = p->child[0].p, n0 = kad_len(q[0]);
	q[1] = p->child[1].p, n1 = kad_len(q[1]);
	if (action == KAD_SYNC_DIM) {
		if (n0 % n1 != 0) return -1;
		kad_sync_dim1(p, q[0]);
	} else if (action == KAD_FORWARD) {
		assert(n0 >= n1);
		memset(p->x, 0, n0 * sizeof(float));
		if (q[0]->x != 0 && q[1]->x != 0)
			for (i = 0; i < n0; i += n1) // TODO: optimize when n1==1
				kad_vec_mul_sum(n1, p->x + i, q[0]->x + i, q[1]->x);
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(q[0]) && q[1]->x)
			for (i = 0; i < n0; i += n1)
				kad_vec_mul_sum(n1, q[0]->g + i, p->g + i, q[1]->x);
		if (kad_is_back(q[1]) && q[0]->x)
			for (i = 0; i < n0; i += n1)
				kad_vec_mul_sum(n1, q[1]->g, p->g + i, q[0]->x + i);
	}
	return 0;
}

int kad_op_cmul(kad_node_t *p, int action)
{
	int n_a_row, n_b_row, n_col, n_a_col, n_b_col;
	kad_node_t *q[2];

	q[0] = p->child[0].p;
	q[1] = p->child[1].p;
	n_a_col = q[0]->n_d == 1? q[0]->d[0] : kad_len(q[0]) / q[0]->d[0];
	n_b_col = q[1]->n_d == 1? q[1]->d[0] : kad_len(q[1]) / q[1]->d[0];
	n_a_row = kad_len(q[0]) / n_a_col, n_b_row = kad_len(q[1]) / n_b_col;
	n_col = n_a_col;
	if (action == KAD_SYNC_DIM) {
		if (n_a_col != n_b_col) return -1;
		p->n_d = 2, p->d[0] = n_a_row, p->d[1] = n_b_row;
	} else if (action == KAD_FORWARD) {
		memset(p->x, 0, n_a_row * n_b_row * sizeof(float));
		if (q[0]->x && q[1]->x)
			kad_sgemm_simple(0, 1, n_a_row, n_b_row, n_col, q[0]->x, q[1]->x, p->x); // Y = X * trans(W)
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(q[0]) && q[1]->x)
			kad_sgemm_simple(0, 0, n_a_row, n_col, n_b_row, p->g, q[1]->x, q[0]->g); // G_x <- G_y * W
		if (kad_is_back(q[1]) && q[0]->x)
			kad_sgemm_simple(1, 0, n_b_row, n_col, n_a_row, p->g, q[0]->x, q[1]->g); // G_w <- trans(G_y) * X
	}
	return 0;
}

int kad_op_matmul(kad_node_t *p, int action)
{
	int n_a_row, n_b_row, n_a_col, n_b_col;
	kad_node_t *q[2];

	q[0] = p->child[0].p;
	q[1] = p->child[1].p;
	n_a_row = q[0]->n_d == 1? 1 : q[0]->d[0];
	n_b_row = q[1]->n_d == 1? 1 : q[1]->d[0];
	n_a_col = kad_len(q[0]) / n_a_row;
	n_b_col = kad_len(q[1]) / n_b_row;
	if (action == KAD_SYNC_DIM) {
		if (n_a_col != n_b_row) return -1;
		p->n_d = 2, p->d[0] = n_a_row, p->d[1] = n_b_col;
	} else if (action == KAD_FORWARD) {
		memset(p->x, 0, n_a_row * n_b_col * sizeof(float));
		if (q[0]->x && q[1]->x)
			kad_sgemm_simple(0, 0, n_a_row, n_b_col, n_a_col, q[0]->x, q[1]->x, p->x); // Y = X * W
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(q[0]) && q[1]->x)
			kad_sgemm_simple(0, 1, n_a_row, n_a_col, n_b_col, p->g, q[1]->x, q[0]->g); // G_x <- G_y * trans(W)
		if (kad_is_back(q[1]) && q[0]->x)
			kad_sgemm_simple(1, 0, n_b_row, n_b_col, n_a_row, q[0]->x, p->g, q[1]->g); // G_y <- trans(A) * G_y
	}
	return 0;
}

int kad_op_norm2(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0].p;
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		p->n_d = 0;
	} else if (action == KAD_FORWARD) {
		p->x[0] = kad_sdot(n, q->x, q->x);
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(q)) {
			float s = 1.0f / n;
			for (i = 0; i < n; ++i)
				q->g[i] += s * p->g[0] * (q->x[i] + q->x[i]);
		}
	}
	return 0;
}

int kad_op_1minus(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0].p;
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_sync_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i) p->x[i] = 1.0f - q->x[i];
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(q))
			kad_saxpy(n, -1.0f, p->g, q->g);
	}
	return 0;
}

/////////// Miscellaneous ///////////

int kad_op_dropout(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0].p;
	assert(p->child[1].p->n_d == 0);
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_sync_dim1(p, q);
	} else if (action == KAD_ALLOC) {
		if (kad_is_back(p->child[0].p))
			p->child[0].t = (float*)realloc(p->child[0].t, n);
	} else if (action == KAD_FORWARD) {
		float r = *p->child[1].p->x, z = 1.0f / (1.0f - r);
		unsigned char *flag = (unsigned char*)p->child[0].t;
		for (i = 0; i < n; ++i) {
			int kept = (kad_drand() >= r);
			p->x[i] = kept? q->x[i] * z : 0.0f;
			if (flag) flag[i] = kept;
		}
	} else if (action == KAD_BACKWARD) {
		unsigned char *flag = (unsigned char*)p->child[0].t;
		if (flag)
			for (i = 0; i < n; ++i)
				if (flag[i]) q->g[i] += p->g[i];
	}
	return 0;
}

int kad_op_split(kad_node_t *p, int action)
{
	kad_node_t *q = p->child[0].p;
	int32_t *aux, n, *range;
	int i, dim, d0, d1;

	assert(p->ptr);
	aux = (int*)p->ptr, dim = aux[0], range = aux + 1;
	if (dim < 0 || dim >= q->n_d) return -1;
	n = kad_len(q);
	for (i = 0, d0 = 1; i < dim; ++i) d0 *= q->d[i];
	for (i = dim + 1, d1 = 1; i < q->n_d; ++i) d1 *= q->d[i];
	if (action == KAD_SYNC_DIM) {
		if (range[0] >= range[1] || range[0] < 0 || range[1] > q->d[dim]) return -1;
		kad_sync_dim1(p, q);
		p->d[dim] = range[1] - range[0];
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < d0; ++i)
			memcpy(&p->x[i * p->d[dim] * d1], &q->x[(i * q->d[dim] + range[0]) * d1], (range[1] - range[0]) * d1 * sizeof(float));
	} else if (action == KAD_BACKWARD && kad_is_back(q)) {
		for (i = 0; i < d0; ++i)
			kad_saxpy((range[1] - range[0]) * d1, 1.0f, &p->g[i * p->d[dim] * d1], &q->g[(i * q->d[dim] + range[0]) * d1]);
	}
	return 0;
}

/////////// Binary and multi-class cross-entropy ///////////

int kad_op_ceb(kad_node_t *p, int action)
{
	static const float tiny = 1e-9f;
	kad_edge_t *e[2];
	int i, n0, n1;

	e[0] = &p->child[0], e[1] = &p->child[1];
	n0 = kad_len(e[0]->p);
	n1 = kad_len(e[1]->p);
	if (action == KAD_SYNC_DIM) {
		if (n0 != n1) return -1;
		p->n_d = 0;
	} else if (action == KAD_ALLOC) {
		assert(kad_is_back(e[0]->p));
		if (kad_is_back(e[0]->p))
			e[0]->t = (float*)realloc(e[0]->t, n0 * sizeof(float));
	} else if (action == KAD_FORWARD) {
		const float *x, *y;
		double s;
		x = e[0]->p->x, y = e[1]->p->x;
		for (i = 0, s = 0.0; i < n0; ++i) {
			float t, y1 = 1.0f - y[i];
			t = 1.0f / (1.0f + expf(-x[i]));
			if (kad_is_back(e[0]->p)) e[0]->t[i] = (t - y[i]) / n0;
			s -= (y[i] == 0.0f? 0.0f : y[i] * logf(t / y[i] + tiny)) + (y1 == 0.0f? 0.0f : y1 * logf((1.0f - t) / y1 + tiny));
		}
		p->x[0] = s / n0;
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(e[0]->p))
			kad_saxpy(n0, p->g[0], e[0]->t, e[0]->p->g);
	}
	return 0;
}

int kad_op_cem(kad_node_t *p, int action)
{
	kad_edge_t *e[2];
	int i, j, n0, n1;

	e[0] = &p->child[0], e[1] = &p->child[1];
	assert(e[0]->p->n_d == 2);
	n0 = kad_len(e[0]->p);
	n1 = kad_len(e[1]->p);
	if (action == KAD_SYNC_DIM) {
		if (n0 != n1) return -1;
		p->n_d = 0;
	} else if (action == KAD_ALLOC) {
		e[0]->t = (float*)realloc(e[0]->t, n0 * sizeof(float));
	} else if (action == KAD_FORWARD) {
		double cost;
		int r = e[0]->p->d[0], c = e[0]->p->d[1];
		for (i = 0; i < n0; ++i) e[0]->t[i] = expf(e[0]->p->x[i]); // FIXME: numerical stability!
		for (j = 0, cost = 0.0; j < r; ++j) {
			const float *x, *y;
			float *p, lsx, sx = 0.0f, sy = 0.0f;
			x = e[0]->p->x + j * c;
			y = e[1]->p->x + j * c;
			p = e[0]->t + j * c;
			for (i = 0; i < c; ++i)
				sx += p[i], sy += y[i];
			assert(sx > 0.0 && sy > 0.0);
			lsx = logf(sx);
			sx = 1.0f / sx, sy = 1.0f / sy;
			for (i = 0; i < c; ++i) {
				float yi = y[i] * sy;
				if (yi != 0.0f) cost += yi * (logf(yi) - (x[i] - lsx));
				p[i] = (p[i] * sx - yi) / r;
			}
		}
		p->x[0] = cost / r;
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(e[0]->p))
			kad_saxpy(n0, p->g[0], e[0]->t, e[0]->p->g);
	}
	return 0;
}

/////////// Activation functions ///////////

int kad_op_sigm(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0].p;
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_sync_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i)
			p->x[i] = 1.0f / (1.0f + expf(-q->x[i]));
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(q)) {
			float s = 1.0f / n;
			for (i = 0; i < n; ++i)
				q->g[i] += s * p->g[i] * (p->x[i] * (1.0f - p->x[i]));
		}
	}
	return 0;
}

int kad_op_tanh(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0].p;
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_sync_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i) {
			if (q->x[i] < -20.0f) p->x[i] = -1.0f;
			else {
				float y;
				y = expf(-2.0f * q->x[i]);
				p->x[i] = (1.0f - y) / (1.0f + y);
			}
		}
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(q))
			for (i = 0; i < n; ++i)
				q->g[i] += p->g[i] * (1.0f - p->x[i] * p->x[i]);
	}
	return 0;
}

int kad_op_relu(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0].p;
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		kad_sync_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i)
			p->x[i] = q->x[i] > 0.0f? q->x[i] : 0.0f;
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(q))
			for (i = 0; i < n; ++i)
				if (q->x[i] > 0.0f)
					q->g[i] += p->g[i];
	}
	return 0;
}

int kad_op_softmax(kad_node_t *p, int action)
{
	int i, j;
	kad_node_t *q = p->child[0].p;
	assert(q->n_d == 2);
	if (action == KAD_SYNC_DIM) {
		kad_sync_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		float t1 = p->n_child >= 2 && p->child[1].p->x? 1.0f / *p->child[1].p->x : 1.0f;
		for (j = 0; j < p->d[0]; ++j) {
			float *x0, *x, s;
			x0 = q->x + j * p->d[1];
			x = p->x + j * p->d[1];
			for (i = 0, s = 0.0f; i < p->d[1]; ++i)
				s += (x[i] = expf(x0[i] * t1));
			s = 1.0f / s;
			for (i = 0; i < p->d[1]; ++i) x[i] *= s;
		}
	}
	return 0;
}

/////////// General pooling operator ///////////

int kad_op_avg(kad_node_t *p, int action)
{
	int i, n;
	float tmp;
	kad_node_t *q;

	assert(p->n_child > 0);
	tmp = 1.0f / p->n_child;
	q = p->child[0].p;
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		for (i = 1; i < p->n_child; ++i)
			if (kad_len(p->child[i].p) != n) return -1;
		kad_sync_dim1(p, q);
	} else if (action == KAD_FORWARD) {
		memcpy(p->x, q->x, n * sizeof(float));
		for (i = 1; i < p->n_child; ++i)
			kad_saxpy(n, 1.0f, p->child[i].p->x, p->x);
		for (i = 0; i < n; ++i) p->x[i] *= tmp;
	} else if (action == KAD_BACKWARD) {
		for (i = 0; i < p->n_child; ++i)
			if (kad_is_back(p->child[i].p))
				kad_saxpy(n, tmp, p->g, p->child[i].p->g);
	}
	return 0;
}

int kad_op_max(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0].p;
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		int *max_j;
		for (i = 1; i < p->n_child; ++i)
			if (kad_len(p->child[i].p) != n) return -1;
		kad_sync_dim1(p, q);
		max_j = (int*)calloc(n, sizeof(int));
		p->child[0].t = (float*)max_j;
	} else if (action == KAD_FORWARD) {
		int j, *max_j = (int*)p->child[0].t;
		memset(max_j, 0, n * sizeof(int));
		memcpy(p->x, q->x, n * sizeof(float));
		for (j = 1; j < p->n_child; ++j)
			for (i = 0, q = p->child[j].p; i < n; ++i)
				if (q->x[i] > p->x[i]) p->x[i] = q->x[i], max_j[i] = j;
	} else if (action == KAD_BACKWARD) {
		int *max_j = (int*)p->child[0].t;
		for (i = 0; i < n; ++i)
			p->child[max_j[i]].p->g[i] += p->g[i];
	}
	return 0;
}

/////////// 2D convolution ///////////

static void conv_rot180(int d0, int d1, float *x) // rotate/reverse a weight martix
{
	int i, j;
	for (i = 0; i < d0; ++i) {
		float tmp, *xi = &x[i * d1];
		for (j = 0; j < d1>>1; ++j)
			tmp = xi[j], xi[j] = xi[d1-1-j], xi[d1-1-j] = tmp; 
	}
}

static void conv2d_move_1to3(int d[4], const float *x, float *y) // convert the NCHW shape to the NHWC shape
{
	int i, j, k, l;
	for (i = 0; i < d[0]; ++i)
		for (j = 0; j < d[1]; ++j)
			for (k = 0; k < d[2]; ++k) {
				int ik = (i * d[2] + k) * d[3], ijk = ((i * d[1] + j) * d[2] + k) * d[3];
				for (l = 0; l < d[3]; ++l)
					y[(ik + l) * d[1] + j] = x[ijk + l];
			}
}

static void conv2d_add_3to1(int d[4], const float *y, float *x) // convert the NHWC shape back to NCHW and add to another NCHW-shaped array
{
	int i, j, k, l;
	for (i = 0; i < d[0]; ++i)
		for (j = 0; j < d[1]; ++j)
			for (k = 0; k < d[2]; ++k) {
				int ik = (i * d[2] + k) * d[3], ijk = ((i * d[1] + j) * d[2] + k) * d[3];
				for (l = 0; l < d[3]; ++l)
					x[ijk + l] += y[(ik + l) * d[1] + j];
			}
}

#define conv_out_size(in_size, aux) (((in_size) - (aux)->kernel_size + (aux)->pad[0] + (aux)->pad[1]) / (aux)->stride + 1)

#define process_row_for(_xx, _ww, _yy, _wn, _pn, _stride, _pad, _t) do { \
	int j, l; \
	if (_stride > 1) { \
		for (l = 0; l < _wn; ++l) { \
			const float *xl = &_xx[l - _pad]; \
			for (j = 0; j < _pn; ++j, xl += _stride) _t[j] = *xl; \
			kad_saxpy(_pn, _ww[l], _t, _yy); \
		} \
	} else for (l = 0; l < _wn; ++l) kad_saxpy(_pn, _ww[l], &_xx[l - _pad], _yy); \
} while (0)

#define process_row_back_x(_xx, _ww, _yy, _wn, _pn, _stride, _pad, _t) do { \
	int j, l; \
	if (_stride > 1) { \
		for (l = 0; l < _wn; ++l) { \
			float *xl = &_xx[l - _pad]; \
			memset(_t, 0, _pn * sizeof(float)); \
			kad_saxpy(_pn, _ww[l], _yy, _t); \
			for (j = 0; j < _pn; ++j, xl += _stride) *xl += _t[j]; \
		} \
	} else for (l = 0; l < _wn; ++l) kad_saxpy(_pn, _ww[l], _yy, &_xx[l - _pad]); \
} while (0)

#define process_row_back_w(_xx, _ww, _yy, _wn, _pn, _stride, _pad, _t) do { \
	int j, l; \
	if (_stride > 1) { \
		for (l = 0; l < _wn; ++l) { \
			const float *xl = &_xx[l - _pad]; \
			for (j = 0; j < _pn; ++j, xl += _stride) _t[j] = *xl; \
			_ww[l] += kad_sdot(_pn, _yy, _t); \
		} \
	} else for (l = 0; l < _wn; ++l) _ww[l] += kad_sdot(_pn, _yy, &_xx[l - _pad]); \
} while (0)

/* Forward and backward passes are implemented with two different algorithms.
 * The first is faster for small kernels with few input channels; otherwise the
 * second algorithm is faster. Both algorithms should produce identical
 * results, up to the precision of "float".
 */
int kad_op_conv2d(kad_node_t *p, int action) // in the number-channel-height-width (NCHW) shape
{
#define conv2d_loop1(_x, _w, _y, _tmp, _row_func) do { /* for the NCHW shape */ \
		int n, c1, c0, i, k, ii; \
		for (n = 0; n < q->d[0]; ++n) /* mini-batch */ \
			for (c1 = 0; c1 < w->d[0]; ++c1) /* output channel */ \
				for (c0 = 0; c0 < w->d[1]; ++c0) /* input channel */ \
					for (k = 0; k < w->d[2]; ++k) { /* kernel row */ \
						float *_ww = &(_w)[((c1 * w->d[1] + c0) * w->d[2] + k) * w->d[3]]; \
						for (i = 0, ii = k - aux[0].pad[0]; i < p->d[2] && ii >= 0 && ii < q->d[2]; ++i, ii += aux[0].stride) { /* output row */ \
							float *_xx = &(_x)[((n * q->d[1] + c0) * q->d[2] + ii) * q->d[3]]; \
							float *_yy = &(_y)[((n * p->d[1] + c1) * p->d[2] + i)  * p->d[3]]; \
							if (x_padded) { \
								memcpy(x_padded + aux[1].pad[0], _xx, q->d[3] * sizeof(float)); \
								_xx = x_padded + aux[1].pad[0]; \
							} \
							_row_func(_xx, _ww, _yy, w->d[3], p->d[3], aux[1].stride, aux[1].pad[0], (_tmp)); \
						} /* ~i */ \
					} /* ~k, c0, c1, n */ \
	} while (0)

#define conv2d_loop2(_x, _w, _y, _code) do { /* for the NHWC shape */ \
		int n, c1, i, j, k, ii, j_skip = aux[1].stride * q->d[1], m = w->d[3] * w->d[1]; \
		for (n = 0; n < q->d[0]; ++n) /* mini-batch */ \
			for (c1 = 0; c1 < w->d[0]; ++c1) /* output channel */ \
				for (k = 0; k < w->d[2]; ++k) { /* kernel row */ \
					float *_ww = &(_w)[(c1 * w->d[2] + k) * m]; \
					for (i = 0, ii = k - aux[0].pad[0]; i < p->d[2] && ii >= 0 && ii < q->d[2]; ++i, ii += aux[0].stride) { /* output and input row */ \
						float *_xx = &(_x)[(n * q->d[2] + ii) * q->d[3] * q->d[1]]; \
						float *_yy = &(_y)[((n * p->d[1] + c1) * p->d[2] + i) * p->d[3]]; \
						if (x_padded) { \
							memcpy(x_padded + aux[1].pad[0] * q->d[1], _xx, q->d[3] * q->d[1] * sizeof(float)); \
							_xx = x_padded; \
						} \
						for (j = 0; j < p->d[3]; ++j, _xx += j_skip, ++_yy) _code; /* output and input column */ \
					} /* ~i */ \
				} /* ~k, c1, n */ \
	} while (0)

	conv_conf_t *aux = (conv_conf_t*)p->ptr;
	kad_node_t *q = p->child[0].p, *w = p->child[1].p;
	float *t = 0, *q1 = 0, *w1 = 0, *x_padded = 0;
	int algo_switch = 0;

	if (action == KAD_FORWARD || action == KAD_BACKWARD) { // allocate working space
		if (w->d[3] * w->d[1] < 16) {
			t = (float*)malloc(p->d[3] * sizeof(float));
			x_padded = aux[1].pad[0] + aux[1].pad[1] > 0? (float*)calloc(q->d[3] + aux[1].pad[0] + aux[1].pad[1], sizeof(float)) : 0;
		} else {
			q1 = (float*)malloc(kad_len(q) * sizeof(float));
			w1 = (float*)malloc(kad_len(w) * sizeof(float));
			x_padded = aux[1].pad[0] + aux[1].pad[1] > 0? (float*)calloc((q->d[3] + aux[1].pad[0] + aux[1].pad[1]) * q->d[1], sizeof(float)) : 0;
			algo_switch = 1;
		}
	}
	if (action == KAD_SYNC_DIM) {
		if (q->n_d != 4 || w->n_d != 4) return -1;
		if (q->d[1] != w->d[1]) return -1; // unmatched input channels
		p->n_d = 4;
		p->d[0] = q->d[0], p->d[1] = w->d[0], p->d[2] = conv_out_size(q->d[2], &aux[0]), p->d[3] = conv_out_size(q->d[3], &aux[1]);
	} else if (action == KAD_FORWARD) {
		conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->x);
		memset(p->x, 0, kad_len(p) * sizeof(float));
		if (!algo_switch) { // this is the first algorithm
			conv2d_loop1(q->x, w->x, p->x, t, process_row_for);
		} else { // this is the second algorithm
			conv2d_move_1to3(q->d, q->x, q1);
			conv2d_move_1to3(w->d, w->x, w1);
			conv2d_loop2(q1, w1, p->x, (*_yy += kad_sdot(m, _ww, _xx)));
		}
		conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->x);
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(p->child[0].p)) { // backprop to the input array
			conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->x);
			if (!algo_switch) {
				conv2d_loop1(q->g, w->x, p->g, t, process_row_back_x);
			} else {
				memset(q1, 0, kad_len(q) * sizeof(float));
				conv2d_move_1to3(w->d, w->x, w1);
				conv2d_loop2(q1, w1, p->g, kad_saxpy(m, *_yy, _ww, _xx));
				conv2d_add_3to1(q->d, q1, q->g);
			}
			conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->x);
		}
		if (kad_is_back(p->child[1].p)) { // backprop to the weight matrix
			conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->g);
			if (!algo_switch) {
				conv2d_loop1(q->x, w->g, p->g, t, process_row_back_w);
			} else {
				conv2d_move_1to3(q->d, q->x, q1);
				memset(w1, 0, kad_len(w) * sizeof(float));
				conv2d_loop2(q1, w1, p->g, kad_saxpy(m, *_yy, _xx, _ww));
				conv2d_add_3to1(w->d, w1, w->g);
			}
			conv_rot180(w->d[0] * w->d[1], w->d[2] * w->d[3], w->g);
		}
	}
	free(t); free(q1); free(w1); free(x_padded);
	return 0;
}

int kad_op_max2d(kad_node_t *p, int action)
{
	conv_conf_t *aux = (conv_conf_t*)p->ptr;
	kad_node_t *q = p->child[0].p;
	if (action == KAD_SYNC_DIM) {
		if (q->n_d != 4) return -1;
		p->n_d = 4;
		p->d[0] = q->d[0], p->d[1] = q->d[1], p->d[2] = conv_out_size(q->d[2], &aux[0]), p->d[3] = conv_out_size(q->d[3], &aux[1]);
	} else if (action == KAD_ALLOC) {
		p->child[0].t = (float*)realloc(p->child[0].t, kad_len(p) * sizeof(int));
	} else if (action == KAD_FORWARD) {
		int rest = 1, len, t, i;
		int *f = (int*)p->child[0].t;
		len = kad_len(p);
		for (i = 0; i < len; ++i) p->x[i] = -FLT_MAX;
		for (i = 0; i < p->n_d - 2; ++i) rest *= p->d[i];
		for (t = 0; t < rest; ++t) {
			int i, j, k, l, p_row = p->d[p->n_d - 2], p_col = p->d[p->n_d - 1];
			for (i = 0; i < p_row; ++i) {
				int u = (t * p_row + i) * p_col;
				for (k = 0; k < aux[0].kernel_size; ++k) {
					int v, v0, v_end, ii = i * aux[0].stride + k - aux[0].pad[0];
					if (ii < 0 || ii >= q->d[p->n_d - 2]) continue;
					v0 = (t * q->d[p->n_d - 2] + ii) * q->d[p->n_d - 1];
					v_end = v0 + q->d[p->n_d - 1];
					for (l = 0; l < aux[1].kernel_size; ++l)
						for (j = 0, v = v0 + (l > aux[1].pad[0]? l - aux[1].pad[0] : 0); j < p_col && v < v_end; ++j, v += aux[1].stride)
							if (p->x[u + j] < q->x[v])
								p->x[u + j] = q->x[v], f[u + j] = v;
				} // ~k
			} // ~i
		}
	} else if (action == KAD_BACKWARD) {
		int i, len, *f = (int*)p->child[0].t;
		len = kad_len(p);
		for (i = 0; i < len; ++i) q->g[f[i]] += p->g[i];
	}
	return 0;
}

/////////// 1D convolution ///////////

static void conv1d_move_1to2(int d[3], const float *x, float *y)
{
	int i, j, k;
	for (k = 0; k < d[0]; ++k)
		for (j = 0; j < d[1]; ++j)
			for (i = 0; i < d[2]; ++i)
				y[(k * d[2] + i) * d[1] + j] = x[(k * d[1] + j) * d[2] + i];
}

static void conv1d_add_2to1(int d[3], const float *y, float *x)
{
	int i, j, k;
	for (k = 0; k < d[0]; ++k)
		for (j = 0; j < d[1]; ++j)
			for (i = 0; i < d[2]; ++i)
				x[(k * d[1] + j) * d[2] + i] += y[(k * d[2] + i) * d[1] + j];
}

int kad_op_conv1d(kad_node_t *p, int action) // in the number-channel-width (NCW) shape
{
#define conv1d_loop1(_x, _w, _y, _tmp, _row_func) do { /* for the NCW shape */ \
		int n, c1, c0; \
		for (n = 0; n < q->d[0]; ++n) /* mini-batch */ \
			for (c1 = 0; c1 < w->d[0]; ++c1) /* output channel */ \
				for (c0 = 0; c0 < w->d[1]; ++c0) { /* input channel */ \
					float *_ww = &(_w)[(c1 * w->d[1] + c0) * w->d[2]]; \
					float *_xx = &(_x)[(n  * q->d[1] + c0) * q->d[2]]; \
					float *_yy = &(_y)[(c1 * p->d[1] + c1) * p->d[2]]; \
					if (x_padded) { \
						memcpy(x_padded + aux->pad[0], _xx, q->d[2] * sizeof(float)); \
						_xx = x_padded + aux->pad[0]; \
					} \
					_row_func(_xx, _ww, _yy, w->d[2], p->d[2], aux->stride, aux->pad[0], (_tmp)); \
				} /* ~c0, c1, n */ \
	} while (0)

#define conv1d_loop2(_x, _w, _y, _code) do { /* for the NWC shape */ \
		int n, c1, j, j_skip = aux->stride * q->d[1], m = w->d[2] * w->d[1]; \
		for (n = 0; n < q->d[0]; ++n) /* mini-batch */ \
			for (c1 = 0; c1 < w->d[0]; ++c1) { /* output channel */ \
				float *_ww = &(_w)[c1 * m]; \
				float *_xx = &(_x)[n * q->d[1] * q->d[2]]; \
				float *_yy = &(_y)[(n * p->d[1] + c1) * p->d[2]]; \
				if (x_padded) { \
					memcpy(x_padded + aux->pad[0] * q->d[1], _xx, q->d[2] * q->d[1] * sizeof(float)); \
					_xx = x_padded; \
				} \
				for (j = 0; j < p->d[2]; ++j, _xx += j_skip, ++_yy) _code; \
			} /* ~c1, n */ \
	} while (0)

	conv_conf_t *aux = (conv_conf_t*)p->ptr;
	kad_node_t *q = p->child[0].p, *w = p->child[1].p;
	float *t = 0, *q1 = 0, *w1 = 0, *x_padded = 0;
	int algo_switch = 0;

	if (action == KAD_FORWARD || action == KAD_BACKWARD) { // allocate working space
		if (w->d[2] * w->d[1] < 32) {
			t = (float*)malloc(p->d[2] * sizeof(float));
			x_padded = aux->pad[0] + aux->pad[1] > 0? (float*)calloc(q->d[2] + aux->pad[0] + aux->pad[1], sizeof(float)) : 0;
		} else {
			q1 = (float*)malloc(kad_len(q) * sizeof(float));
			w1 = (float*)malloc(kad_len(w) * sizeof(float));
			x_padded = aux->pad[0] + aux->pad[1] > 0? (float*)calloc((q->d[2] + aux->pad[0] + aux->pad[1]) * q->d[1], sizeof(float)) : 0;
			algo_switch = 1;
		}
	}
	if (action == KAD_SYNC_DIM) {
		if (q->n_d != 3 || w->n_d != 3) return -1;
		if (q->d[1] != w->d[1]) return -1; // unmatched input channels
		p->n_d = 3;
		p->d[0] = q->d[0], p->d[1] = w->d[0], p->d[2] = conv_out_size(q->d[2], aux);
	} else if (action == KAD_FORWARD) {
		conv_rot180(w->d[0] * w->d[1], w->d[2], w->x);
		memset(p->x, 0, kad_len(p) * sizeof(float));
		if (!algo_switch) { // this is the first algorithm
			conv1d_loop1(q->x, w->x, p->x, t, process_row_for);
		} else { // this is the second algorithm
			conv1d_move_1to2(q->d, q->x, q1);
			conv1d_move_1to2(w->d, w->x, w1);
			conv1d_loop2(q1, w1, p->x, (*_yy += kad_sdot(m, _ww, _xx)));
		}
		conv_rot180(w->d[0] * w->d[1], w->d[2], w->x);
	} else if (action == KAD_BACKWARD) {
		if (kad_is_back(p->child[0].p)) { // backprop to the input array
			conv_rot180(w->d[0] * w->d[1], w->d[2], w->x);
			if (!algo_switch) {
				conv1d_loop1(q->g, w->x, p->g, t, process_row_back_x);
			} else {
				memset(q1, 0, kad_len(q) * sizeof(float));
				conv1d_move_1to2(w->d, w->x, w1);
				conv1d_loop2(q1, w1, p->g, kad_saxpy(m, *_yy, _ww, _xx));
				conv1d_add_2to1(q->d, q1, q->g);
			}
			conv_rot180(w->d[0] * w->d[1], w->d[2], w->x);
		}
		if (kad_is_back(p->child[1].p)) { // backprop to the weight matrix
			conv_rot180(w->d[0] * w->d[1], w->d[2], w->g);
			if (!algo_switch) {
				conv1d_loop1(q->x, w->g, p->g, t, process_row_back_w);
			} else {
				conv1d_move_1to2(q->d, q->x, q1);
				memset(w1, 0, kad_len(w) * sizeof(float));
				conv1d_loop2(q1, w1, p->g, kad_saxpy(m, *_yy, _xx, _ww));
				conv1d_add_2to1(w->d, w1, w->g);
			}
			conv_rot180(w->d[0] * w->d[1], w->d[2], w->g);
		}
	}
	free(t); free(q1); free(w1); free(x_padded);
	return 0;
}

int kad_op_max1d(kad_node_t *p, int action)
{
	conv_conf_t *aux = (conv_conf_t*)p->ptr;
	kad_node_t *q = p->child[0].p;
	if (action == KAD_SYNC_DIM) {
		if (q->n_d != 3) return -1;
		p->n_d = 3;
		p->d[0] = q->d[0], p->d[1] = q->d[1], p->d[2] = conv_out_size(q->d[2], aux);
	} else if (action == KAD_ALLOC) {
		p->child[0].t = (float*)realloc(p->child[0].t, kad_len(p) * sizeof(int));
	} else if (action == KAD_FORWARD) {
		int rest = 1, len, t, i;
		int *f = (int*)p->child[0].t;
		len = kad_len(p);
		for (i = 0; i < len; ++i) p->x[i] = -FLT_MAX;
		for (i = 0; i < p->n_d - 1; ++i) rest *= p->d[i];
		for (t = 0; t < rest; ++t) {
			int j, l, p_width = p->d[p->n_d - 1];
			int u = t * p_width, v, v0 = t * q->d[p->n_d - 1], v_end = v0 + q->d[p->n_d - 1];
			for (l = 0; l < aux->kernel_size; ++l)
				for (j = 0, v = v0 + (l > aux->pad[0]? l - aux->pad[0] : 0); j < p_width && v < v_end; ++j, v += aux->stride)
					if (p->x[u + j] < q->x[v])
						p->x[u + j] = q->x[v], f[u + j] = v;
		}
	} else if (action == KAD_BACKWARD) {
		int i, len, *f = (int*)p->child[0].t;
		len = kad_len(p);
		for (i = 0; i < len; ++i) q->g[f[i]] += p->g[i];
	}
	return 0;
}

/////////// List of operators ///////////

kad_op_f kad_op_list[KAD_MAX_OP] = {
	0,
	kad_op_add,     // 1:  element-wise addition
	kad_op_mul,     // 2:  element-wise multiplication
	kad_op_cmul,    // 3:  column multiplication
	kad_op_ceb,     // 4:  binary cross-entroy for sigmoid activation
	kad_op_norm2,   // 5:  L2-norm
	kad_op_sigm,    // 6:  sigmoid
	kad_op_tanh,    // 7:  tanh
	kad_op_relu,    // 8:  ReLU
	kad_op_matmul,  // 9:  matrix multiplication
	kad_op_avg,     // 10: general average pooling (not for ConvNet)
	kad_op_1minus,  // 11: 1-x
	kad_op_cem,     // 12: multi-class cross-entropy for softmax activation
	kad_op_softmax, // 13: softmax without temperature
	kad_op_softmax, // 14: softmax with temperature
	kad_op_dropout, // 15: dropout
	kad_op_conv2d,  // 16: 2D convolution
	kad_op_max2d,   // 17: 2D max pooling (for 2D ConvNet)
	kad_op_conv1d,  // 18: 1D convolution
	kad_op_max1d,   // 19: 1D max pooling (for 1D ConvNet)
	kad_op_split,   // 20: split data at a dimension
	kad_op_max      // 21: general max pooling
};

/**************************
 *** Debugging routines ***
 **************************/

void kad_trap_fe(void)
{
#ifdef __SSE__
	_MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~(_MM_MASK_INVALID | _MM_MASK_DIV_ZERO));
#endif
}

void kad_print_graph(FILE *fp, int n, kad_node_t **v)
{
	static const char *op[] = { 0, "add", "mul", "cmul", "ceb", "norm2", "sigm", "tanh", "relu", 0, "avg", "1minus", "cem", "softmax2", "softmax",
								"dropout", "conv2d", "max2d", "conv1d", "max1d", "subset", "max" };
	int i, j;
	for (i = 0; i < n; ++i) v[i]->tmp = i;
	for (i = 0; i < n; ++i) {
		kad_node_t *p = v[i];
		fprintf(stderr, "%d\t%d\t", i, p->ext_label);
		if (p->pre) fprintf(fp, "%d\t", p->pre->tmp);
		else fprintf(fp, ".\t");
		fputs("[", fp);
		for (j = 0; j < p->n_d; ++j) {
			if (j) fputc(',', fp);
			fprintf(fp, "%d", p->d[j]);
		}
		fprintf(fp, "]\t");
		if (p->n_child) {
			fprintf(fp, "%s(", op[p->op]);
			for (j = 0; j < p->n_child; ++j) {
				if (j) fputc(',', fp);
				fprintf(fp, "$%d", p->child[j].p->tmp);
			}
			fprintf(fp, ")");
		} else fprintf(fp, "%s", kad_is_back(p)? "var" : "par");
		fputc('\n', fp);
	}
	for (i = 0; i < n; ++i) v[i]->tmp = 0;
}

static void kad_add_delta(int n, kad_node_t **a, float c, float *delta)
{
	int i, k;
	for (i = k = 0; i < n; ++i)
		if (kad_is_var(a[i])) {
			kad_saxpy(kad_len(a[i]), c, &delta[k], a[i]->x);
			k += kad_len(a[i]);
		}
}

void kad_check_grad(int n, kad_node_t **a, int from)
{
	const float eps = 1e-5, rel = 1e-7 / eps;
	int i, k, n_var;
	float *g0, *delta, f0, f_minus, f_plus, s0, s1, rel_err, p_m_err;
	n_var = kad_n_var(n, a);
	g0 = (float*)calloc(n_var, sizeof(float));
	f0 = *kad_eval_at(n, a, from);
	kad_grad(n, a, from);
	for (i = k = 0; i < n; ++i)
		if (kad_is_var(a[i])) {
			memcpy(&g0[k], a[i]->g, kad_len(a[i]) * sizeof(float));
			k += kad_len(a[i]);
		}
	delta = (float*)calloc(n_var, sizeof(float));
	for (k = 0; k < n_var; ++k) delta[k] = drand48() * eps;
	kad_add_delta(n, a, 1.0f, delta);
	f_plus = *kad_eval_at(n, a, from);
	kad_add_delta(n, a, -2.0f, delta);
	f_minus = *kad_eval_at(n, a, from);
	kad_add_delta(n, a, 1.0f, delta);
	s0 = kad_sdot(n_var, g0, delta);
	s1 = .5 * (f_plus - f_minus);
	fprintf(stderr, "Gradient check -- %g <=> %g @ %g -- ", s0/eps, s1/eps, f0);
	if (fabs(s1) >= rel * eps) {
		rel_err = fabs(fabs(s0) - fabs(s1)) / (fabs(s0) + fabs(s1));
		p_m_err = fabs(f_plus + f_minus - 2.0f * f0) / fabs(f_plus - f_minus);
		fprintf(stderr, "rel_err:%g p_m_err:%g -- ", rel_err, p_m_err);
		if (rel_err >= rel && rel_err > p_m_err) fprintf(stderr, "failed\n");
		else fprintf(stderr, "passed\n");
	} else fprintf(stderr, "skipped\n");
	free(delta); free(g0);
}
