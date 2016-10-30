#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "kautodiff.h"

/**********************
 * Graph construction *
 **********************/

static inline kad_node_t *kad_new_core(int op, int n_child)
{
	kad_node_t *s;
	s = (kad_node_t*)calloc(1, sizeof(kad_node_t));
	s->op = op, s->n_child = n_child;
	if (s->n_child) s->child = (kad_edge_t*)calloc(s->n_child, sizeof(kad_edge_t));
	return s;
}

kad_node_t *kad_par(int n_row, int n_col, const float *x)
{
	kad_node_t *s;
	s = kad_new_core(0, 0);
	s->n_row = n_row, s->n_col = n_col;
	s->_.cx = x;
	return s;
}

kad_node_t *kad_var(int n_row, int n_col, const float *x, float *d)
{
	kad_node_t *s;
	s = kad_par(n_row, n_col, x);
	s->d = d, s->to_back = 1;
	return s;
}

static inline kad_node_t *kad_op2_core(int op, kad_node_t *x, kad_node_t *y)
{
	kad_node_t *s;
	s = kad_new_core(op, 2);
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
	s = kad_new_core(op, 1);
	s->child[0].p = x;
	kad_op_list[op](s, KAD_SYNC_DIM);
	return s;
}

#define KAD_FUNC_OP2(fname, op) kad_node_t *fname(kad_node_t *x, kad_node_t *y) { return kad_op2_core((op), x, y); }

KAD_FUNC_OP2(kad_add, 1)
KAD_FUNC_OP2(kad_sub, 2)
KAD_FUNC_OP2(kad_mul, 3)
KAD_FUNC_OP2(kad_mtmul, 4)
KAD_FUNC_OP2(kad_ce2, 5)

#define KAD_FUNC_OP1(fname, op) kad_node_t *fname(kad_node_t *x) { return kad_op1_core((op), x); }

KAD_FUNC_OP1(kad_norm2, 6)
KAD_FUNC_OP1(kad_sigm, 7)
KAD_FUNC_OP1(kad_tanh, 8)
KAD_FUNC_OP1(kad_relu, 9)

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

typedef struct kad_node_t *kad_node_p;

// IMPORTANT: kad_node_t::tmp MUST BE set to zero
kad_node_t **kad_compile(kad_node_t *root, int *n_node)
{
	int i, j;
	kvec_t(kad_node_p) stack = {0,0,0}, a = {0,0,0};

	// generate kad_node_t::tmp
	kv_push(kad_node_p, stack, root);
	while (stack.n) {
		kad_node_t *p = kv_pop(stack);
		for (i = 0; i < p->n_child; ++i) {
			kad_node_t *q = p->child[i].p;
			if (q->tmp == 0) kv_push(kad_node_p, stack, q);
			++q->tmp;
		}
	}
	// topological sorting (Kahn's algorithm)
	kv_push(kad_node_p, stack, root);
	while (stack.n) {
		kad_node_t *p = kv_pop(stack);
		kv_push(kad_node_p, a, p);
		for (i = 0; i < p->n_child; ++i)
			if (--p->child[i].p->tmp == 0)
				kv_push(kad_node_p, stack, p->child[i].p);
	}
	free(stack.a);
	// reverse a
	for (i = 0; i < a.n>>1; ++i) {
		kad_node_p t;
		t = a.a[i], a.a[i] = a.a[a.n-1-i], a.a[a.n-1-i] = t;
	}
	// check cycles
	for (i = 0; i < a.n; ++i) assert(a.a[i]->tmp == 0); // if the graph is constructed with kad_add() etc, there should be no cycles
	// set kad_node_t::to_back and allocate
	for (i = 0; i < a.n; ++i) {
		kad_node_p p = a.a[i];
		if (p->n_child == 0) continue;
		p->_.x = (float*)realloc(p->_.x, p->n_row * p->n_col * sizeof(float));
		for (j = 0; j < p->n_child; ++j)
			if (p->child[j].p->to_back) break;
		if (j < p->n_child) {
			p->to_back = 1;
			p->d = (float*)realloc(p->d, p->n_row * p->n_col * sizeof(float));
			kad_op_list[p->op](p, KAD_ALLOC);
		}
	}
	*n_node = a.n;
	return a.a;
}

void kad_free(int n, kad_node_t **a)
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

float kad_eval(int n, kad_node_t **a, int cal_grad)
{
	int i;
	float f;
	assert(n > 0);
	for (i = 0; i < n; ++i) // forward pass
		if (a[i]->n_child)
			kad_op_list[a[i]->op](a[i], KAD_FORWARD);
	f = a[n-1]->_.x[0];
	if (cal_grad) {
		assert(a[n-1]->n_row == 1 && a[n-1]->n_col == 1);
		for (i = 0; i < n; ++i) // set all grandients to zero
			if (a[i]->to_back)
				memset(a[i]->d, 0, a[i]->n_row * a[i]->n_col * sizeof(float));
		for (i = n - 1, a[i]->d[0] = 1.0f; i >= 0; --i) // backprop
			if (a[i]->n_child)
				kad_op_list[a[i]->op](a[i], KAD_BACKWARD);
	}
	return f;
}

/***********************
 * Load and save graph *
 ***********************/

int kad_write(FILE *fp, int n_node, kad_node_t **node)
{
	int i, j;
	fwrite(&n_node, sizeof(int), 1, fp);
	for (i = 0; i < n_node; ++i) node[i]->tmp = i;
	for (i = 0; i < n_node; ++i) {
		kad_node_t *p = node[i];
		fwrite(&p->n_child, sizeof(short), 1, fp);
		if (p->n_child) {
			fwrite(&p->op, sizeof(int), 1, fp);
			for (j = 0; j < p->n_child; ++j)
				fwrite(&p->child[j].p->tmp, sizeof(int), 1, fp);
		} else {
			fwrite(&p->label, sizeof(int), 1, fp);
			fwrite(&p->n_row, sizeof(int), 1, fp);
			fwrite(&p->n_col, sizeof(int), 1, fp);
			fwrite(&p->to_back, sizeof(short), 1, fp);
		}
	}
	for (i = 0; i < n_node; ++i) node[i]->tmp = i;
	return 0;
}

kad_node_t **kad_read(FILE *fp, int *_n_node)
{
	int i, j, n_node;
	kad_node_t **node;
	fread(&n_node, sizeof(int), 1, fp);
	node = (kad_node_t**)malloc(n_node * sizeof(kad_node_t*));
	for (i = 0; i < n_node; ++i) {
		kad_node_t *p;
		p = node[i] = (kad_node_t*)calloc(1, sizeof(kad_node_t));
		fread(&p->n_child, sizeof(short), 1, fp);
		if (p->n_child) {
			p->child = (kad_edge_t*)calloc(1, sizeof(kad_edge_t));
			fread(&p->op, sizeof(int), 1, fp);
			for (j = 0; j < p->n_child; ++j) {
				int k;
				fread(&k, sizeof(int), 1, fp);
				assert(k < i);
				p->child[j].p = node[k];
			}
			kad_op_list[p->op](p, KAD_SYNC_DIM);
		} else {
			fread(&p->label, sizeof(int), 1, fp);
			fread(&p->n_row, sizeof(int), 1, fp);
			fread(&p->n_col, sizeof(int), 1, fp);
			fread(&p->to_back, sizeof(short), 1, fp);
		}
	}
	*_n_node = n_node;
	return node;
}

/*********************
 * Vector operations *
 *********************/

#ifdef __SSE__
#include <xmmintrin.h>

float kad_sdot(int n, const float *x, const float *y)
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
void kad_saxpy(int n, float a, const float *x, float *y)
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
float kad_sdot(int n, const float *x, const float *y) // BLAS sdot
{
	int i;
	float s = 0.;
	for (i = 0; i < n; ++i) s += x[i] * y[i];
	return s;
}
void kad_saxpy(int n, float a, const float *x, float *y) // BLAS saxpy
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

void kad_mat_mtmul(int n_col, int n_a_row, const float *a, int n_b_row, const float *b, float *c) // C = A * B^T
{
	static const int x = 16;
	int i, j;
	memset(c, 0, n_a_row * n_b_row * sizeof(float));
	for (i = 0; i < n_a_row; i += x) {
		for (j = 0; j < n_b_row; j += x) {
			int ii, ie = n_a_row < i + x? n_a_row : i + x;
			int jj, je = n_b_row < j + x? n_b_row : j + x;
			for (ii = i; ii < ie; ++ii) {
				const float *aii = a + ii * n_col, *bjj;
				float *cii = c + ii * n_b_row;
				for (jj = j, bjj = b + j * n_col; jj < je; ++jj, bjj += n_col)
					cii[jj] += kad_sdot(n_col, aii, bjj);
			}
		}
	}
}

/*************
 * Operators *
 *************/

int kad_op_add(kad_node_t *p, int action)
{
	int i, n = p->n_row * p->n_col;
	kad_edge_t *e[2];

	e[0] = &p->child[0];
	e[1] = &p->child[1];
	if (action == KAD_SYNC_DIM) {
		if ((e[0]->p->n_row != e[1]->p->n_row && e[1]->p->n_row != 1) || e[0]->p->n_col != e[1]->p->n_col) return -1;
		p->n_row = e[0]->p->n_row, p->n_col = e[0]->p->n_col;
	} else if (action == KAD_FORWARD) {
		memcpy(p->_.x, e[0]->p->_.x, n * sizeof(float));
		if (e[1]->p->n_row == 1)
			for (i = 0; i < e[0]->p->n_row; ++i)
				kad_saxpy(e[0]->p->n_col, 1.0f, e[1]->p->_.x, &p->_.x[i*e[0]->p->n_col]);
		else kad_saxpy(n, 1.0f, e[1]->p->_.x, p->_.x);
	} else if (action == KAD_BACKWARD) {
		if (e[0]->p->to_back) kad_saxpy(n, 1.0f, p->d, e[0]->p->d);
		if (e[1]->p->to_back) {
			if (e[1]->p->n_row == 1)
				for (i = 0; i < e[0]->p->n_row; ++i)
					kad_saxpy(e[0]->p->n_col, 1.0f, &p->d[i*e[0]->p->n_col], e[1]->p->d);
			else kad_saxpy(n, 1.0f, p->d, e[1]->p->d);
		}
	}
	return 0;
}

int kad_op_sub(kad_node_t *p, int action)
{
	int n = p->n_row * p->n_col;
	kad_edge_t *e[2];

	e[0] = &p->child[0];
	e[1] = &p->child[1];
	if (action == KAD_SYNC_DIM) {
		if (e[0]->p->n_row != e[1]->p->n_row || e[0]->p->n_col != e[1]->p->n_col) return -1;
		p->n_row = e[0]->p->n_row, p->n_col = e[0]->p->n_col;
	} else if (action == KAD_FORWARD) {
		memcpy(p->_.x, e[0]->p->_.x, n * sizeof(float));
		kad_saxpy(n, -1.0f, e[1]->p->_.x, p->_.x);
	} else if (action == KAD_BACKWARD) {
		if (e[0]->p->to_back) kad_saxpy(n, 1.0f, p->d, e[0]->p->d);
		if (e[1]->p->to_back) kad_saxpy(n, -1.0f, p->d, e[1]->p->d);
	}
	return 0;
}

int kad_op_mul(kad_node_t *p, int action)
{
	int i, n = p->n_row * p->n_col;
	kad_edge_t *e[2];

	e[0] = &p->child[0];
	e[1] = &p->child[1];
	if (action == KAD_SYNC_DIM) {
		if (e[0]->p->n_row != e[1]->p->n_row || e[0]->p->n_col != e[1]->p->n_col) return -1;
		p->n_row = e[0]->p->n_row, p->n_col = e[0]->p->n_col;
	} else if (action == KAD_FORWARD) {
		memset(p->_.x, 0, n * sizeof(float));
		kad_vec_mul_sum(n, p->_.x, e[0]->p->_.x, e[1]->p->_.x);
	} else if (action == KAD_BACKWARD) {
		for (i = 0; i < 2; ++i)
			if (e[i]->p->to_back)
				kad_vec_mul_sum(n, e[i]->p->d, p->d, e[!i]->p->_.x);
	}
	return 0;
}

int kad_op_mtmul(kad_node_t *p, int action)
{
	kad_edge_t *e[2];

	e[0] = &p->child[0];
	e[1] = &p->child[1];
	assert(e[0]->p->to_back == 0);
	if (action == KAD_SYNC_DIM) {
		if (e[0]->p->n_col != e[1]->p->n_col) return -1;
		p->n_row = e[0]->p->n_row, p->n_col = e[1]->p->n_row;
	} else if (action == KAD_FORWARD) {
		kad_mat_mtmul(e[0]->p->n_col, e[0]->p->n_row, e[0]->p->_.x, e[1]->p->n_row, e[1]->p->_.x, p->_.x);
	} else if (action == KAD_BACKWARD) {
		if (e[1]->p->to_back) {
			int i, j, n_col = e[0]->p->n_col;
			for (i = 0; i < e[0]->p->n_row; ++i)
				for (j = 0; j < e[1]->p->n_row; ++j)
					kad_saxpy(n_col, p->d[i * e[1]->p->n_row + j], e[0]->p->_.x + i * n_col, e[1]->p->d + j * n_col);
		}
	}
	return 0;
}

int kad_op_ce2(kad_node_t *p, int action)
{
	kad_edge_t *e[2];
	int i, n;

	assert(p->child[1].p->to_back == 0); // child[1] is the true; we don't backprop this
	e[0] = &p->child[0], e[1] = &p->child[1];
	n = e[0]->p->n_row * e[0]->p->n_col;
	if (action == KAD_SYNC_DIM) {
		p->n_row = p->n_col = 1;
	} else if (action == KAD_ALLOC) {
		if (e[0]->p->to_back)
			e[0]->t = (float*)realloc(e[0]->t, n * sizeof(float));
	} else if (action == KAD_FORWARD) {
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
	} else if (action == KAD_BACKWARD) {
		if (e[0]->p->to_back)
			kad_saxpy(n, p->d[0], e[0]->t, e[0]->p->d);
	}
	return 0;
}

int kad_op_norm2(kad_node_t *p, int action)
{
	kad_edge_t *e = &p->child[0];
	int i, n = e->p->n_row * e->p->n_col;
	if (action == KAD_SYNC_DIM) {
		p->n_row = p->n_col = 1;
	} else if (action == KAD_FORWARD) {
		p->_.x[0] = kad_sdot(n, e->p->_.x, e->p->_.x);
	} else if (action == KAD_BACKWARD) {
		if (e->p->to_back)
			for (i = 0; i < n; ++i)
				e->p->d[i] += p->d[i] * (e->p->_.x[i] + e->p->_.x[i]);
	}
	return 0;
}

int kad_op_sigm(kad_node_t *p, int action)
{
	int i, n = p->n_row * p->n_col;
	kad_edge_t *e = &p->child[0];
	if (action == KAD_SYNC_DIM) {
		p->n_row = e->p->n_row, p->n_col = e->p->n_col;
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i)
			p->_.x[i] = 1.0f / (1.0f + expf(e->p->_.x[i]));
	} else if (action == KAD_BACKWARD) {
		if (e->p->to_back)
			for (i = 0; i < n; ++i)
				e->p->d[i] += p->d[i] * (p->_.x[i] * (1.0f - p->_.x[i]));
	}
	return 0;
}

int kad_op_tanh(kad_node_t *p, int action)
{
	int i, n = p->n_row * p->n_col;
	kad_edge_t *e = &p->child[0];
	if (action == KAD_SYNC_DIM) {
		p->n_row = e->p->n_row, p->n_col = e->p->n_col;
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i) {
			float y;
			y = expf(-2.0f * e->p->_.x[i]);
			p->_.x[i] = (1.0f - y) / (1.0f + y);
		}
	} else if (action == KAD_BACKWARD) {
		if (e->p->to_back)
			for (i = 0; i < n; ++i)
				e->p->d[i] += p->d[i] * (1.0f - p->_.x[i] * p->_.x[i]);
	}
	return 0;
}

int kad_op_relu(kad_node_t *p, int action)
{
	int i, n = p->n_row * p->n_col;
	kad_edge_t *e = &p->child[0];
	if (action == KAD_SYNC_DIM) {
		p->n_row = e->p->n_row, p->n_col = e->p->n_col;
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i)
			p->_.x[i] = e->p->_.x[i] > 0.0f? e->p->_.x[i] : 0.0f;
	} else if (action == KAD_BACKWARD) {
		if (e->p->to_back)
			for (i = 0; i < n; ++i)
				if (e->p->_.x[i] > 0.0f)
					e->p->d[i] += p->d[i];
	}
	return 0;
}

kad_op_f kad_op_list[] = {
	0,
	kad_op_add,
	kad_op_sub,
	kad_op_mul,
	kad_op_mtmul,
	kad_op_ce2,
	kad_op_norm2,
	kad_op_sigm,
	kad_op_tanh,
	kad_op_relu,
	0
};
