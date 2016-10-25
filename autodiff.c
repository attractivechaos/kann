#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "autodiff.h"

/**********************
 * Graph construction *
 **********************/

static inline ad_node_t *ad_new_core(int op, int n_row, int n_col, int n_child, const float *x, float *d)
{
	ad_node_t *s;
	s = (ad_node_t*)calloc(1, sizeof(ad_node_t));
	s->op = op, s->n_row = n_row, s->n_col = n_col, s->n_child = n_child, s->_.cx = x, s->d = d;
	if (s->n_child) s->child = (ad_edge_t*)calloc(s->n_child, sizeof(ad_edge_t));
	if (d) s->to_back = 1;
	return s;
}

ad_node_t *ad_var(int n_row, int n_col, const float *x, float *d) { return ad_new_core(0, n_row, n_col, 0, x, d); }
ad_node_t *ad_param(int n_row, int n_col, const float *x) { return ad_new_core(0, n_row, n_col, 0, x, 0); }

static inline ad_node_t *ad_op2_core(int op, int n_row, int n_col, ad_node_t *x, ad_node_t *y)
{
	ad_node_t *s;
	s = ad_new_core(op, n_row, n_col, 2, 0, 0);
	s->child[0].p = x, s->child[1].p = y;
	return s;
}

#define AD_FUNC_OP2(fname, op, cond, _row, _col) \
	ad_node_t *fname(ad_node_t *x, ad_node_t *y) { return (cond)? 0 : ad_op2_core((op), (_row), (_col), x, y); }

AD_FUNC_OP2(ad_add, 1, (x->n_row != y->n_row || x->n_col != y->n_col), x->n_row, x->n_col)
AD_FUNC_OP2(ad_sub, 2, (x->n_row != y->n_row || x->n_col != y->n_col), x->n_row, x->n_col)
AD_FUNC_OP2(ad_mul, 3, (x->n_row != y->n_row || x->n_col != y->n_col), x->n_row, x->n_col)
AD_FUNC_OP2(ad_mtmul, 4, (x->n_col != y->n_col), x->n_row, y->n_row)
//AD_FUNC_OP2(ad_smul, 5, (x->n_row == 1 && x->n_col == 1), y->n_row, y->n_col)
//AD_FUNC_OP2(ad_dot, 6, (x->n_row != y->n_row || x->n_col != y->n_col), 1, x->n_col)
//AD_FUNC_OP2(ad_div, 7, (x->n_row != y->n_row || x->n_col != y->n_col), x->n_row, x->n_col)
AD_FUNC_OP2(ad_ce2, 8, (x->n_row != y->n_row || x->n_col != y->n_col), 1, y->n_col)

static inline ad_node_t *ad_op1_core(int op, int n_row, int n_col, ad_node_t *x)
{
	ad_node_t *s;
	s = ad_new_core(op, n_row, n_col, 1, 0, 0);
	s->child[0].p = x;
	return s;
}

#define AD_FUNC_OP1(fname, op, _row, _col) \
	ad_node_t *fname(ad_node_t *x) { return ad_op1_core((op), (_row), (_col), x); }

AD_FUNC_OP1(ad_norm2, 9, 1, x->n_col)
AD_FUNC_OP1(ad_sigm, 10, x->n_row, x->n_col)
AD_FUNC_OP1(ad_tanh, 11, x->n_row, x->n_col)

/*******************
 * Graph traversal *
 *******************/

#define kvec_t(type) struct { size_t n, m; type *a; }

#define kv_pop(v) ((v).a[--(v).n])

#define kv_push(type, v, x) do { \
		if ((v).n == (v).m) { \
			(v).m = (v).m? (v).m<<1 : 2; \
			(v).a = (type*)realloc((v).a, sizeof(type) * (v).m); \
		} \
		(v).a[(v).n++] = (x); \
	} while (0)

typedef struct ad_node_t *ad_node_p;

ad_node_t **ad_compile(ad_node_t *root, int *n_node)
{
	int i, j;
	kvec_t(ad_node_p) stack = {0,0,0}, a = {0,0,0};

	// generate ad_node_t::cnt
	kv_push(ad_node_p, stack, root);
	while (stack.n) {
		ad_node_t *p = kv_pop(stack);
		for (i = 0; i < p->n_child; ++i) {
			ad_node_t *q = p->child[i].p;
			if (q->cnt == 0) kv_push(ad_node_p, stack, q);
			++q->cnt;
		}
	}
	// topological sorting (Kahn's algorithm)
	kv_push(ad_node_p, stack, root);
	while (stack.n) {
		ad_node_t *p = kv_pop(stack);
		kv_push(ad_node_p, a, p);
		for (i = 0; i < p->n_child; ++i)
			if (--p->child[i].p->cnt == 0)
				kv_push(ad_node_p, stack, p->child[i].p);
	}
	free(stack.a);
	// reverse a
	for (i = 0; i < a.n>>1; ++i) {
		ad_node_p t;
		t = a.a[i], a.a[i] = a.a[a.n-1-i], a.a[a.n-1-i] = t;
	}
	// check cycles
	for (i = 0; i < a.n; ++i)
		if (a.a[i]->cnt != 0) break;
	if (i < a.n) {
		*n_node = 0;
		free(a.a);
		return 0;
	}
	// decide which edges to backward
	for (i = 0; i < a.n; ++i) {
		ad_node_p p = a.a[i];
		for (j = 0; j < p->n_child; ++j)
			if (p->child[j].p->to_back) break;
		if (j < p->n_child) p->to_back = 1;
	}
	*n_node = a.n;
	return a.a;
}

void ad_free(int n, ad_node_t **a)
{
	int i, j;
	for (i = 0; i < n; ++i) {
		for (j = 0; j < a[i]->n_child; ++j)
			free(a[i]->child[j].z);
		if (a[i]->n_child) {
			free(a[i]->_.x);
			free(a[i]->d);
		}
		free(a[i]->child);
	}
	free(a);
}

/*****************************
 * Automatic differentiation *
 *****************************/

void ad_vec_saxpy(int n, float a, const float *x, float *y); // BLAS saxpy
void ad_vec_elem_mul(int n, const float *x, const float *y, float *z);

typedef void (*ad_op_f)(struct ad_node_t*);

static ad_op_f ad_op_list[]; // actual operators are defined and implemented at the bottom of this source file

float ad_forward(int n, ad_node_t **a)
{
	int i;
	assert(n > 0 && a[n-1]->n_row == 1 && a[n-1]->n_col == 1);
	for (i = 0; i < n; ++i) {
		ad_node_t *p = a[i];
		if (p->n_child == 0) continue;
		ad_op_list[p->op](p);
	}
	return a[n-1]->_.x[0];
}

void ad_backward(int n, ad_node_t **a)
{
	int i, j;
	assert(n > 0 && a[n-1]->n_row == 1 && a[n-1]->n_col == 1);
	// allocate the gradient array if necessary and zero
	for (i = 0; i < n; ++i) 
		if (a[i]->to_back && a[i]->n_child) {
			a[i]->d = (float*)realloc(a[i]->d, a[i]->n_row * a[i]->n_col * sizeof(float));
			memset(a[i]->d, 0, a[i]->n_row * a[i]->n_col * sizeof(float));
		}
	// backprop
	a[n-1]->d[0] = 1.0f;
	for (i = n - 1; i >= 0; --i) {
		ad_node_t *p = a[i];
		if (p->n_child == 0) continue;
		for (j = 0; j < p->n_child; ++j) {
			ad_edge_t *e = &p->child[j];
			if (!e->p->to_back) continue;
			if (e->dtype == AD_DT_IDEN) {
				ad_vec_saxpy(p->n_row, 1.0f, p->d, e->p->d);
			} else if (e->dtype == AD_DT_NEGIDEN) {
				ad_vec_saxpy(p->n_row, -1.0f, p->d, e->p->d);
			} else if (e->dtype == AD_DT_DIAG) {
				ad_vec_elem_mul(p->n_row, p->d, e->z, e->p->d);
			} else if (e->dtype == AD_DT_OUTMAT) {
				int k, l;
				for (k = 0; k < p->n_row; ++k)
					for (l = 0; l < e->p->n_row; ++l)
						ad_vec_saxpy(e->p->n_col, p->d[k * p->n_col + l], e->z + l * e->p->n_col, e->p->d + k * e->p->n_col);
			} else if (e->dtype == AD_DT_VEC) {
				assert(p->n_row == 1);
				ad_vec_saxpy(e->p->n_row, p->d[0], e->z, e->p->d);
			} else {
				assert(0);
			}
		}
	}
}

float ad_eval(int n, ad_node_t **a)
{
	float fret;
	fret = ad_forward(n, a);
	ad_backward(n, a);
	return fret;
}

/*********************
 * Vector operations *
 *********************/

#ifdef __SSE__
#include <xmmintrin.h>

float ad_vec_sdot(int n, const float *x, const float *y)
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
void ad_vec_saxpy(int n, float a, const float *x, float *y)
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
void ad_vec_saxpy(int n, float a, const float *x, float *y) // BLAS saxpy
{
	int i;
	for (i = 0; i < n; ++i) y[i] += a * x[i];
}
float ad_vec_sdot(int n, const float *x, const float *y) // BLAS sdot
{
	int i;
	float s = 0.;
	for (i = 0; i < n; ++i) s += x[i] * y[i];
	return s;
}
#endif

void ad_vec_elem_mul(int n, const float *x, const float *y, float *z)
{
	int i;
	for (i = 0; i < n; ++i) z[i] += x[i] * y[i];
}

void ad_mat_mtmul(int n_col, int n_a_row, const float *a, int n_b_row, const float *b, float *c) // C = A * B^T
{
	static const int x = 16;
	int i, j;
	for (i = 0; i < n_a_row; i += x) {
		for (j = 0; j < n_b_row; j += x) {
			int ii, ie = n_a_row < i + x? n_a_row : i + x;
			int jj, je = n_b_row < j + x? n_b_row : j + x;
			for (ii = i; ii < ie; ++ii)
				for (jj = j; jj < je; ++jj)
					c[ii*n_b_row+jj] += ad_vec_sdot(n_col, &a[ii*n_col], &b[jj*n_col]);
		}
	}
}

/*************
 * Operators *
 *************/

void ad_op_add(ad_node_t *p)
{
	int n = p->n_row * p->n_col;
	ad_edge_t *e[2];
	e[0] = &p->child[0];
	e[1] = &p->child[1];
	p->_.x = (float*)realloc(p->_.x, n * sizeof(float));
	memcpy(p->_.x, e[0]->p->_.x, n * sizeof(float));
	ad_vec_saxpy(n, 1.0f, e[1]->p->_.x, p->_.x);
	if (e[0]->p->to_back) e[0]->dtype = AD_DT_IDEN;
	if (e[1]->p->to_back) e[1]->dtype = AD_DT_IDEN;
}

void ad_op_sub(ad_node_t *p)
{
	int n = p->n_row * p->n_col;
	ad_edge_t *e[2];
	e[0] = &p->child[0];
	e[1] = &p->child[1];
	p->_.x = (float*)realloc(p->_.x, n * sizeof(float));
	memcpy(p->_.x, e[0]->p->_.x, n * sizeof(float));
	ad_vec_saxpy(n, -1.0f, e[1]->p->_.x, p->_.x);
	if (e[0]->p->to_back) e[0]->dtype = AD_DT_IDEN;
	if (e[1]->p->to_back) e[1]->dtype = AD_DT_NEGIDEN;
}

void ad_op_mul(ad_node_t *p)
{
	int n = p->n_row * p->n_col;
	ad_edge_t *e[2];
	e[0] = &p->child[0];
	e[1] = &p->child[1];
	p->_.x = (float*)realloc(p->_.x, n * sizeof(float));
	memset(p->_.x, 0, n * sizeof(float));
	ad_vec_elem_mul(n, e[0]->p->_.x, e[1]->p->_.x, p->_.x);
	if (e[0]->p->to_back) {
		e[0]->dtype = AD_DT_DIAG;
		e[0]->z = (float*)realloc(e[0]->z, n * sizeof(float));
		memcpy(e[0]->z, e[1]->p->_.x, n * sizeof(float));
	}
	if (e[1]->p->to_back) {
		e[1]->dtype = AD_DT_DIAG;
		e[1]->z = (float*)realloc(e[1]->z, n * sizeof(float));
		memcpy(e[1]->z, e[0]->p->_.x, n * sizeof(float));
	}
}

void ad_op_mtmul(ad_node_t *p)
{
	int n = p->n_row * p->n_col;
	ad_edge_t *e[2];
	e[0] = &p->child[0];
	e[1] = &p->child[1];
	assert(e[0]->p->to_back == 0);
	p->_.x = (float*)realloc(p->_.x, n * sizeof(float));
	ad_mat_mtmul(e[0]->p->n_col, e[0]->p->n_row, e[0]->p->_.x, e[1]->p->n_row, e[1]->p->_.x, p->_.x);
	if (e[1]->p->to_back) {
		n = e[0]->p->n_row * e[0]->p->n_col;
		e[1]->dtype = AD_DT_OUTMAT;
		e[1]->z = (float*)realloc(e[1]->z, n * sizeof(float));
		memcpy(e[1]->z, e[0]->p->_.x, n * sizeof(float));
	}
}

void ad_op_smul(ad_node_t *p) {}
void ad_op_dot(ad_node_t *p) {}
void ad_op_div(ad_node_t *p) {}

void ad_op_ce2(ad_node_t *p)
{
	ad_edge_t *e[2];
	int i, n;
	const float *x, *y;
	double s;

	assert(p->child[1].p->to_back == 0); // child[1] is the true; we don't backprop this
	e[0] = &p->child[0], e[1] = &p->child[1];
	n = e[0]->p->n_row * e[0]->p->n_col;
	x = e[0]->p->_.x, y = e[1]->p->_.x;
	p->_.x = (float*)realloc(p->_.x, sizeof(float));
	if (e[0]->p->to_back) {
		e[0]->dtype = AD_DT_VEC;
		e[0]->z = (float*)realloc(e[0]->z, n * sizeof(float));
	}
	for (i = 0, s = 0.0; i < n; ++i) {
		float t;
		t = 1.0f + expf(-x[i]);
		if (e[0]->p->to_back) e[0]->z[i] = 1.0f / t - y[i];
		t = logf(t);
		if (y[i] != 0.0f) s += y[i] * t;
		if (1.0f - y[i] != 0.0f) s += (1.0f - y[i]) * (x[i] + t);
	}
	p->_.x[0] = s / e[0]->p->n_row;
}

void ad_op_norm2(ad_node_t *p)
{
	ad_edge_t *e = &p->child[0];
	int n = e->p->n_row * e->p->n_col;
	p->_.x = (float*)realloc(p->_.x, sizeof(float));
	p->_.x[0] = ad_vec_sdot(n, e->p->_.x, e->p->_.x) / e->p->n_row;
	if (e->p->to_back) {
		e->dtype = AD_DT_VEC;
		e->z = (float*)realloc(e->z, n * sizeof(float));
		memcpy(e->z, e->p->_.x, n * sizeof(float));
		ad_vec_saxpy(n, 1.0f, e->p->_.x, e->z);
	}
}

void ad_op_sigm(ad_node_t *p)
{
	int i, n = p->n_row * p->n_col;
	ad_edge_t *e;
	e = &p->child[0];
	if (e->p->to_back) {
		e->dtype = AD_DT_DIAG;
		e->z = (float*)realloc(e->z, n * sizeof(float));
	}
	p->_.x = (float*)realloc(p->_.x, n * sizeof(float));
	for (i = 0; i < n; ++i) {
		float y, x = e->p->_.x[i];
		p->_.x[i] = y = 1.0f / (1.0f + expf(x));
		if (e->z) e->z[i] = y * (1.0f - y);
	}
}

void ad_op_tanh(ad_node_t *p)
{
	int i, n = p->n_row * p->n_col;
	ad_edge_t *e;
	e = &p->child[0];
	if (e->p->to_back) {
		e->dtype = AD_DT_DIAG;
		e->z = (float*)realloc(e->z, n * sizeof(float));
	}
	p->_.x = (float*)realloc(p->_.x, n * sizeof(float));
	for (i = 0; i < n; ++i) {
		float y, x = e->p->_.x[i];
		y = expf(2.0f * x);
		p->_.x[i] = y = (1.0f - y) / (1.0f + y);
		if (e->z) e->z[i] = 1.0f - y * y;
	}
}

static ad_op_f ad_op_list[] = {
	0,
	ad_op_add,     // 1: addition
	ad_op_sub,     // 2: subtraction
	ad_op_mul,     // 3: element-wise product
	ad_op_mtmul,   // 4: matrix product
	ad_op_smul,    // 5: scalar-to-matrix product
	ad_op_dot,     // 6: vector dot/inner product
	ad_op_div,     // 7: element-wise division
	ad_op_ce2,     // 8: binary cross-entropy: F(x,y) = \sum_i -y_i*log(f(x_i)) - (1-y_i)*log(1-f(x_i)); f() is sigmoid
	ad_op_norm2,   // 9: ||x|| = x^T x
	ad_op_sigm,    // 10: element-wise sigmoind function
	ad_op_tanh     // 11: tanh
};
