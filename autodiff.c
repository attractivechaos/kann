#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "autodiff.h"

/**********************
 * Graph construction *
 **********************/

static inline ad_node_t *ad_new_core(int op, int n_child)
{
	ad_node_t *s;
	s = (ad_node_t*)calloc(1, sizeof(ad_node_t));
	s->op = op, s->n_child = n_child;
	if (s->n_child) s->child = (ad_edge_t*)calloc(s->n_child, sizeof(ad_edge_t));
	return s;
}

ad_node_t *ad_par(int n_row, int n_col, const float *x)
{
	ad_node_t *s;
	s = ad_new_core(0, 0);
	s->n_row = n_row, s->n_col = n_col;
	s->_.cx = x;
	return s;
}

ad_node_t *ad_var(int n_row, int n_col, const float *x, float *d)
{
	ad_node_t *s;
	s = ad_par(n_row, n_col, x);
	s->d = d, s->to_back = 1;
	return s;
}

static inline ad_node_t *ad_op2_core(int op, ad_node_t *x, ad_node_t *y)
{
	ad_node_t *s;
	s = ad_new_core(op, 2);
	s->child[0].p = x, s->child[1].p = y;
	if (ad_op_list[op](s, AD_SYNCDIM) < 0) {
		free(s->child); free(s);
		return 0;
	}
	return s;
}

static inline ad_node_t *ad_op1_core(int op, ad_node_t *x)
{
	ad_node_t *s;
	s = ad_new_core(op, 1);
	s->child[0].p = x;
	ad_op_list[op](s, AD_SYNCDIM);
	return s;
}

#define AD_FUNC_OP2(fname, op) ad_node_t *fname(ad_node_t *x, ad_node_t *y) { return ad_op2_core((op), x, y); }

AD_FUNC_OP2(ad_add, 1)
AD_FUNC_OP2(ad_sub, 2)
AD_FUNC_OP2(ad_mul, 3)
AD_FUNC_OP2(ad_mtmul, 4)
AD_FUNC_OP2(ad_smul, 5)
AD_FUNC_OP2(ad_ce2, 6)

#define AD_FUNC_OP1(fname, op) ad_node_t *fname(ad_node_t *x) { return ad_op1_core((op), x); }

AD_FUNC_OP1(ad_norm2, 7)
AD_FUNC_OP1(ad_sigm, 8)
AD_FUNC_OP1(ad_tanh, 9)

/*****************************
 * Automatic differentiation *
 *****************************/

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

// IMPORTANT: ad_node_t::tmp MUST BE set to zero
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
			++q->tmp;
		}
	}
	// topological sorting (Kahn's algorithm)
	kv_push(ad_node_p, stack, root);
	while (stack.n) {
		ad_node_t *p = kv_pop(stack);
		kv_push(ad_node_p, a, p);
		for (i = 0; i < p->n_child; ++i)
			if (--p->child[i].p->tmp == 0)
				kv_push(ad_node_p, stack, p->child[i].p);
	}
	free(stack.a);
	// reverse a
	for (i = 0; i < a.n>>1; ++i) {
		ad_node_p t;
		t = a.a[i], a.a[i] = a.a[a.n-1-i], a.a[a.n-1-i] = t;
	}
	// check cycles
	for (i = 0; i < a.n; ++i) assert(a.a[i]->tmp == 0); // if the graph is constructed with ad_add() etc, there should be no cycles
	// set ad_node_t::to_back and allocate
	for (i = 0; i < a.n; ++i) {
		ad_node_p p = a.a[i];
		if (p->n_child == 0) continue;
		p->_.x = (float*)realloc(p->_.x, p->n_row * p->n_col * sizeof(float));
		for (j = 0; j < p->n_child; ++j)
			if (p->child[j].p->to_back) break;
		if (j < p->n_child) {
			p->to_back = 1;
			p->d = (float*)realloc(p->d, p->n_row * p->n_col * sizeof(float));
			ad_op_list[p->op](p, AD_ALLOC);
		}
	}
	*n_node = a.n;
	return a.a;
}

void ad_free(int n, ad_node_t **a)
{
	int i, j;
	for (i = 0; i < n; ++i) {
		for (j = 0; j < a[i]->n_child; ++j)
			free(a[i]->child[j].t);
		if (a[i]->n_child) {
			free(a[i]->_.x);
			free(a[i]->d);
		}
		free(a[i]->child);
	}
	free(a);
}

float ad_eval(int n, ad_node_t **a)
{
	int i;
	float f;
	assert(n > 0 && a[n-1]->n_row == 1 && a[n-1]->n_col == 1);
	for (i = 0; i < n; ++i) // forward pass
		if (a[i]->n_child)
			ad_op_list[a[i]->op](a[i], AD_FORWARD);
	f = a[n-1]->_.x[0];
	a[n-1]->d[0] = 1.0f;
	for (i = n - 1; i >= 0; --i) // backprop
		if (a[i]->n_child)
			ad_op_list[a[i]->op](a[i], AD_BACKWARD);
	return f;
}

/*********************
 * Vector operations *
 *********************/

#ifdef __SSE__
#include <xmmintrin.h>

float ad_sdot(int n, const float *x, const float *y)
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
void ad_saxpy(int n, float a, const float *x, float *y)
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
void ad_vec_mul(int n, float *x, const float *y)
{
	int i, n8 = n>>3<<3;
	for (i = 0; i < n8; i += 8) {
		__m128 vx1, vx2, vy1, vy2, vt1, vt2;
		vx1 = _mm_loadu_ps(&x[i]);
		vx2 = _mm_loadu_ps(&x[i+4]);
		vy1 = _mm_loadu_ps(&y[i]);
		vy2 = _mm_loadu_ps(&y[i+4]);
		vt1 = _mm_mul_ps(vx1, vy1);
		vt2 = _mm_mul_ps(vx2, vy2);
		_mm_storeu_ps(&x[i], vt1);
		_mm_storeu_ps(&x[i+4], vt2);
	}
	for (; i < n; ++i) x[i] *= y[i];
}
#else
float ad_sdot(int n, const float *x, const float *y) // BLAS sdot
{
	int i;
	float s = 0.;
	for (i = 0; i < n; ++i) s += x[i] * y[i];
	return s;
}
void ad_saxpy(int n, float a, const float *x, float *y) // BLAS saxpy
{
	int i;
	for (i = 0; i < n; ++i) y[i] += a * x[i];
}
void ad_vec_mul(int n, float *a, const float *b)
{
	int i;
	for (i = 0; i < n; ++i) a[i] *= b[i];
}
#endif

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
					c[ii*n_b_row+jj] += ad_sdot(n_col, &a[ii*n_col], &b[jj*n_col]);
		}
	}
}

/*************
 * Operators *
 *************/

int ad_op_add(ad_node_t *p, int action)
{
	int n = p->n_row * p->n_col;
	ad_edge_t *e[2];

	e[0] = &p->child[0];
	e[1] = &p->child[1];
	if (action == AD_SYNCDIM) {
		if (e[0]->p->n_row != e[1]->p->n_row || e[0]->p->n_col != e[1]->p->n_col) return -1;
		p->n_row = e[0]->p->n_row, p->n_col = e[0]->p->n_col;
	} else if (action == AD_FORWARD) {
		memcpy(p->_.x, e[0]->p->_.x, n * sizeof(float));
		ad_saxpy(n, 1.0f, e[1]->p->_.x, p->_.x);
	} else if (action == AD_BACKWARD) {
		if (e[0]->p->to_back) memcpy(e[0]->p->d, p->d, n * sizeof(float));
		if (e[1]->p->to_back) memcpy(e[1]->p->d, p->d, n * sizeof(float));
	}
	return 0;
}

int ad_op_sub(ad_node_t *p, int action)
{
	int n = p->n_row * p->n_col;
	ad_edge_t *e[2];

	e[0] = &p->child[0];
	e[1] = &p->child[1];
	if (action == AD_SYNCDIM) {
		if (e[0]->p->n_row != e[1]->p->n_row || e[0]->p->n_col != e[1]->p->n_col) return -1;
		p->n_row = e[0]->p->n_row, p->n_col = e[0]->p->n_col;
	} else if (action == AD_FORWARD) {
		memcpy(p->_.x, e[0]->p->_.x, n * sizeof(float));
		ad_saxpy(n, -1.0f, e[1]->p->_.x, p->_.x);
	} else if (action == AD_BACKWARD) {
		if (e[0]->p->to_back) memcpy(e[0]->p->d, p->d, n * sizeof(float));
		if (e[1]->p->to_back) {
			int i;
			memcpy(e[1]->p->d, p->d, n * sizeof(float));
			for (i = 0; i < n; ++i) e[1]->p->d[i] = -e[1]->p->d[i];
		}
	}
	return 0;
}

int ad_op_mul(ad_node_t *p, int action)
{
	int i, n = p->n_row * p->n_col;
	ad_edge_t *e[2];

	e[0] = &p->child[0];
	e[1] = &p->child[1];
	if (action == AD_SYNCDIM) {
		if (e[0]->p->n_row != e[1]->p->n_row || e[0]->p->n_col != e[1]->p->n_col) return -1;
		p->n_row = e[0]->p->n_row, p->n_col = e[0]->p->n_col;
	} else if (action == AD_FORWARD) {
		memcpy(p->_.x, e[0]->p->_.x, n * sizeof(float));
		ad_vec_mul(n, p->_.x, e[1]->p->_.x);
	} else if (action == AD_BACKWARD) {
		for (i = 0; i < 2; ++i)
			if (e[i]->p->to_back) {
				memcpy(e[i]->p->d, p->d, n * sizeof(float));
				ad_vec_mul(n, e[i]->p->d, e[!i]->p->_.x);
			}
	}
	return 0;
}

int ad_op_smul(ad_node_t *p, int action)
{
	int n = p->n_row * p->n_col;
	ad_edge_t *e[2];

	e[0] = &p->child[0];
	e[1] = &p->child[1];
	assert(e[0]->p->to_back == 0);
	if (action == AD_SYNCDIM) {
		if (e[0]->p->n_row != 1 || e[0]->p->n_col != 1) return -1;
		p->n_row = e[1]->p->n_row, p->n_col = e[1]->p->n_col;
	} else if (action == AD_FORWARD) {
		memset(p->_.x, 0, n * sizeof(float));
		ad_saxpy(n, e[0]->p->_.x[0], e[1]->p->_.x, p->_.x);
	} else if (action == AD_BACKWARD) {
		if (e[1]->p->to_back) {
			memset(e[1]->p->d, 0, n * sizeof(float));
			ad_saxpy(n, e[1]->p->_.x[0], p->d, e[1]->p->d);
		}
	}
	return 0;
}

int ad_op_mtmul(ad_node_t *p, int action)
{
	int n = p->n_row * p->n_col;
	ad_edge_t *e[2];

	e[0] = &p->child[0];
	e[1] = &p->child[1];
	assert(e[0]->p->to_back == 0);
	if (action == AD_SYNCDIM) {
		if (e[0]->p->n_col != e[1]->p->n_col) return -1;
		p->n_row = e[0]->p->n_row, p->n_col = e[1]->p->n_row;
	} else if (action == AD_FORWARD) {
		memset(p->_.x, 0, n * sizeof(float));
		ad_mat_mtmul(e[0]->p->n_col, e[0]->p->n_row, e[0]->p->_.x, e[1]->p->n_row, e[1]->p->_.x, p->_.x);
	} else if (action == AD_BACKWARD) {
		if (e[1]->p->to_back) {
			int i, j, n_col = e[0]->p->n_col;
			for (i = 0; i < e[0]->p->n_row; ++i)
				for (j = 0; j < e[1]->p->n_row; ++j)
					ad_saxpy(n_col, p->d[i * e[1]->p->n_row + j], e[0]->p->_.x + i * n_col, e[1]->p->d + j * n_col);
		}
	}
	return 0;
}

int ad_op_ce2(ad_node_t *p, int action)
{
	ad_edge_t *e[2];
	int i, n;

	assert(p->child[1].p->to_back == 0); // child[1] is the true; we don't backprop this
	e[0] = &p->child[0], e[1] = &p->child[1];
	n = e[0]->p->n_row * e[0]->p->n_col;
	if (action == AD_SYNCDIM) {
		p->n_row = p->n_col = 1;
	} else if (action == AD_ALLOC) {
		if (e[0]->p->to_back)
			e[0]->t = (float*)realloc(e[0]->t, n * sizeof(float));
	} else if (action == AD_FORWARD) {
		const float *x, *y;
		double s;
		x = e[0]->p->_.x, y = e[1]->p->_.x;
		for (i = 0, s = 0.0; i < n; ++i) {
			float t;
			t = 1.0f + expf(-x[i]);
			if (e[0]->p->to_back) e[0]->t[i] = 1.0f / t - y[i];
			t = logf(t);
			if (y[i] != 0.0f) s += y[i] * t;
			if (1.0f - y[i] != 0.0f) s += (1.0f - y[i]) * (x[i] + t);
		}
		p->_.x[0] = s / e[0]->p->n_row;
	} else if (action == AD_BACKWARD) {
		if (e[0]->p->to_back) {
			memset(e[0]->p->d, 0, n * sizeof(float));
			ad_saxpy(n, p->d[0], e[0]->t, e[0]->p->d);
		}
	}
	return 0;
}

int ad_op_norm2(ad_node_t *p, int action)
{
	ad_edge_t *e = &p->child[0];
	int n = e->p->n_row * e->p->n_col;
	if (action == AD_SYNCDIM) {
		p->n_row = p->n_col = 1;
	} else if (action == AD_FORWARD) {
		p->_.x[0] = ad_sdot(n, e->p->_.x, e->p->_.x);
	} else if (action == AD_BACKWARD) {
		if (e->p->to_back) {
			memcpy(e->p->d, e->p->_.x, n * sizeof(float));
			ad_saxpy(n, 1.0f, e->p->_.x, e->p->d);
		}
	}
	return 0;
}

int ad_op_sigm(ad_node_t *p, int action)
{
	int i, n = p->n_row * p->n_col;
	ad_edge_t *e = &p->child[0];
	if (action == AD_SYNCDIM) {
		p->n_row = e->p->n_row, p->n_col = e->p->n_col;
	} else if (action == AD_FORWARD) {
		for (i = 0; i < n; ++i)
			p->_.x[i] = 1.0f / (1.0f + expf(e->p->_.x[i]));
	} else if (action == AD_BACKWARD) {
		if (e->p->to_back)
			for (i = 0; i < n; ++i)
				e->p->d[i] = p->_.x[i] * (1.0f - p->_.x[i]);
	}
	return 0;
}

int ad_op_tanh(ad_node_t *p, int action)
{
	int i, n = p->n_row * p->n_col;
	ad_edge_t *e = &p->child[0];
	if (action == AD_SYNCDIM) {
		p->n_row = e->p->n_row, p->n_col = e->p->n_col;
	} else if (action == AD_FORWARD) {
		for (i = 0; i < n; ++i) {
			float y;
			y = expf(-2.0f * e->p->_.x[i]);
			p->_.x[i] = (1.0f - y) / (1.0f + y);
		}
	} else if (action == AD_BACKWARD) {
		if (e->p->to_back)
			for (i = 0; i < n; ++i)
				e->p->d[i] = 1.0f - p->_.x[i] * p->_.x[i];
	}
	return 0;
}

ad_op_f ad_op_list[] = {
	0,
	ad_op_add,
	ad_op_sub,
	ad_op_mul,
	ad_op_mtmul,
	ad_op_smul,
	ad_op_ce2,
	ad_op_norm2,
	ad_op_sigm,
	ad_op_tanh,
	0
};
