#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <stdarg.h>
#include <math.h>
#include "kautodiff.h"

/**********************
 * Graph construction *
 **********************/

static inline kad_node_t *kad_new_core(int n_d, int op, int n_child)
{
	kad_node_t *s;
	s = (kad_node_t*)calloc(1, sizeof(kad_node_t));
	s->n_d = n_d, s->op = op, s->n_child = n_child;
	if (s->n_child) s->child = (kad_edge_t*)calloc(s->n_child, sizeof(kad_edge_t));
	return s;
}

kad_node_t *kad_par(const float *x, int n_d, ...)
{
	int i;
	kad_node_t *p;
	va_list ap;
	va_start(ap, n_d);
	p = kad_new_core(n_d, 0, 0);
	for (i = 0; i < n_d; ++i)
		p->d[i] = va_arg(ap, int);
	p->_.cx = x;
	va_end(ap);
	return p;
}

kad_node_t *kad_var(const float *x, float *g, int n_d, ...)
{
	int i;
	kad_node_t *p;
	va_list ap;
	va_start(ap, n_d);
	p = kad_new_core(n_d, 0, 0);
	for (i = 0; i < n_d; ++i)
		p->d[i] = va_arg(ap, int);
	p->_.cx = x, p->g = g, p->to_back = 1;
	va_end(ap);
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
KAD_FUNC_OP2(kad_ce2, 4)

#define KAD_FUNC_OP1(fname, op) kad_node_t *fname(kad_node_t *x) { return kad_op1_core((op), x); }

KAD_FUNC_OP1(kad_norm2, 5)
KAD_FUNC_OP1(kad_sigm, 6)
KAD_FUNC_OP1(kad_tanh, 7)
KAD_FUNC_OP1(kad_relu, 8)

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
		p->_.x = (float*)realloc(p->_.x, kad_len(p) * sizeof(float));
		for (j = 0; j < p->n_child; ++j)
			if (p->child[j].p->to_back) break;
		if (j < p->n_child) {
			p->to_back = 1;
			p->g = (float*)realloc(p->g, kad_len(p) * sizeof(float));
			kad_op_list[p->op](p, KAD_ALLOC);
		}
	}
	*n_node = a.n;
	return a.a;
}

void kad_free_node(kad_node_t *p)
{
	int j;
	for (j = 0; j < p->n_child; ++j)
		free(p->child[j].t);
	if (p->n_child) {
		free(p->_.x);
		free(p->g);
	}
	free(p->child);
	free(p);
}

void kad_free(int n, kad_node_t **a)
{
	int i;
	for (i = 0; i < n; ++i) kad_free_node(a[i]);
	free(a);
}

float kad_eval(int n, kad_node_t **a, int cal_grad)
{
	int i;
	float f;
	assert(n > 0);
	for (i = 0; i < n; ++i) // forward pass
		if (a[i]->n_child) kad_for1(a[i]);
	f = a[n-1]->_.x[0];
	if (cal_grad) {
		assert(a[n-1]->n_d == 0);
		for (i = 0; i < n; ++i) // set all grandients to zero
			if (a[i]->to_back)
				memset(a[i]->g, 0, kad_len(a[i]) * sizeof(float));
		for (i = n - 1, a[i]->g[0] = 1.0f; i >= 0; --i) // backprop
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
			fwrite(&p->n_d, sizeof(short), 1, fp);
			if (p->n_d) fwrite(p->d, sizeof(int), p->n_d, fp);
			fwrite(&p->label, sizeof(int), 1, fp);
			fwrite(&p->to_back, sizeof(short), 1, fp);
		}
	}
	for (i = 0; i < n_node; ++i) node[i]->tmp = 0;
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
			fread(&p->n_d, sizeof(short), 1, fp);
			if (p->n_d) fread(p->d, sizeof(int), p->n_d, fp);
			fread(&p->label, sizeof(int), 1, fp);
			fread(&p->to_back, sizeof(short), 1, fp);
		}
	}
	*_n_node = n_node;
	return node;
}

void kad_debug(FILE *fp, int n, kad_node_t **v)
{
	static const char *op[] = { "", "add", "mul", "cmul", "ce2", "norm2", "sigm", "tanh", "relu" };
	int i, j;
	for (i = 0; i < n; ++i) v[i]->tmp = i;
	for (i = 0; i < n; ++i) {
		kad_node_t *p = v[i];
		fprintf(stderr, "%d\t", i);
		if (p->n_child) {
			fprintf(fp, "%s(", op[p->op]);
			for (j = 0; j < p->n_child; ++j) {
				if (j) fputc(',', fp);
				fprintf(fp, "$%d", p->child[j].p->tmp);
			}
			fputc(')', fp);
		} else fprintf(fp, "%c|%d", p->to_back? 'v' : 'p', p->label);
		fputs("\t[", fp);
		for (j = 0; j < p->n_d; ++j) {
			if (j) fputc(',', fp);
			fprintf(fp, "%d", p->d[j]);
		}
		fputc(']', fp);
		fputc('\n', fp);
	}
	for (i = 0; i < n; ++i) v[i]->tmp = 0;
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

void kad_mat_cmul(int n_col, int n_a_row, const float *a, int n_b_row, const float *b, float *c) // C = A * B^T
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
	int i, n0, n1;
	kad_node_t *q[2];

	q[0] = p->child[0].p, n0 = kad_len(q[0]);
	q[1] = p->child[1].p, n1 = kad_len(q[1]);
	if (action == KAD_SYNC_DIM) {
		if (n0 % n1 != 0) return -1;
		p->n_d = q[0]->n_d;
		if (p->n_d) memcpy(p->d, q[0]->d, p->n_d * sizeof(int));
	} else if (action == KAD_FORWARD) {
		memcpy(p->_.x, q[0]->_.x, n0 * sizeof(float));
		for (i = 0; i < n0; i += n1)
			kad_saxpy(n1, 1.0f, q[1]->_.x, p->_.x + i);
	} else if (action == KAD_BACKWARD) {
		if (q[0]->to_back) kad_saxpy(n0, 1.0f, p->g, q[0]->g);
		if (q[1]->to_back)
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
		p->n_d = q[0]->n_d;
		if (p->n_d) memcpy(p->d, q[0]->d, p->n_d * sizeof(int));
	} else if (action == KAD_FORWARD) {
		memset(p->_.x, 0, n0 * sizeof(float));
		for (i = 0; i < n0; i += n1) // TODO: optimize when n1==1
			kad_vec_mul_sum(n1, p->_.x + i, q[0]->_.x + i, q[1]->_.x);
	} else if (action == KAD_BACKWARD) {
		if (q[0]->to_back)
			for (i = 0; i < n0; i += n1)
				kad_vec_mul_sum(n1, q[0]->g + i, p->g + i, q[1]->_.x);
		if (q[1]->to_back)
			for (i = 0; i < n0; i += n1)
				kad_vec_mul_sum(n1, q[1]->g, p->g + i, q[0]->_.x + i);
	}
	return 0;
}

int kad_op_cmul(kad_node_t *p, int action)
{
	int i, j, n_a_row, n_b_row, n_col;
	kad_node_t *q[2];

	q[0] = p->child[0].p;
	q[1] = p->child[1].p;
	if (q[0]->n_d == 1 && q[1]->n_d == 2)      n_a_row = 1, n_b_row = q[1]->d[0], n_col = q[0]->d[0];
	else if (q[0]->n_d == 2 && q[1]->n_d == 1) n_a_row = q[0]->d[0], n_b_row = 1, n_col = q[1]->d[0];
	else if (q[0]->n_d == 2 && q[1]->n_d == 2) n_a_row = q[0]->d[0], n_b_row = q[1]->d[0], n_col = q[0]->d[1];
	else abort();
	if (action == KAD_SYNC_DIM) {
		if (q[0]->n_d == 1 && q[1]->n_d == 2) {
			if (q[0]->d[0] != q[1]->d[1]) return -1;
			p->n_d = 1, p->d[1] = q[0]->d[0];
		} else if (q[0]->n_d == 2 && q[1]->n_d == 1) {
			if (q[0]->d[1] != q[1]->d[0]) return -1;
			p->n_d = 1, p->d[1] = q[1]->d[0];
		} else if (q[0]->n_d == 2 && q[1]->n_d == 2) {
			if (q[0]->d[1] != q[1]->d[1]) return -1;
			p->n_d = 2, p->d[0] = q[0]->d[0], p->d[1] = q[1]->d[0];
		} else return -1;
	} else if (action == KAD_FORWARD) {
		kad_mat_cmul(n_col, n_a_row, q[0]->_.x, n_b_row, q[1]->_.x, p->_.x);
	} else if (action == KAD_BACKWARD) {
		if (q[0]->to_back) // TODO: is this correct?
			for (j = 0; j < n_b_row; ++j)
				for (i = 0; i < n_a_row; ++i)
					kad_saxpy(n_col, p->g[i * n_b_row + j], q[1]->_.x + j * n_col, q[0]->g + i * n_col);
		if (q[1]->to_back)
			for (i = 0; i < n_a_row; ++i)
				for (j = 0; j < n_b_row; ++j)
					kad_saxpy(n_col, p->g[i * n_b_row + j], q[0]->_.x + i * n_col, q[1]->g + j * n_col);
	}
	return 0;
}

int kad_op_ce2(kad_node_t *p, int action)
{
	kad_edge_t *e[2];
	int i, n0, n1;

	e[0] = &p->child[0], e[1] = &p->child[1];
	assert(e[1]->p->to_back == 0); // child[1] is the true; we don't backprop this
	n0 = kad_len(e[0]->p);
	n1 = kad_len(e[1]->p);
	if (action == KAD_SYNC_DIM) {
		if (n0 != n1) return -1;
		p->n_d = 0;
	} else if (action == KAD_ALLOC) {
		if (e[0]->p->to_back)
			e[0]->t = (float*)realloc(e[0]->t, n0 * sizeof(float));
	} else if (action == KAD_FORWARD) {
		const float *x, *y;
		double s;
		x = e[0]->p->_.x, y = e[1]->p->_.x;
		for (i = 0, s = 0.0; i < n0; ++i) {
			float t;
			t = 1.0f + expf(-x[i]);
			if (e[0]->p->to_back) e[0]->t[i] = (1.0f / t - y[i]) / n0;
			t = x[i] < -30.0f? -x[i] : logf(t);
			if (y[i] != 0.0f) s += y[i] * t;
			if (1.0f - y[i] != 0.0f) s += (1.0f - y[i]) * (x[i] + t);
		}
		p->_.x[0] = s / n0;
	} else if (action == KAD_BACKWARD) {
		if (e[0]->p->to_back)
			kad_saxpy(n0, p->g[0], e[0]->t, e[0]->p->g);
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
		p->_.x[0] = kad_sdot(n, q->_.x, q->_.x);
	} else if (action == KAD_BACKWARD) {
		if (q->to_back) {
			float s = 1.0f / n;
			for (i = 0; i < n; ++i)
				q->g[i] += s * p->g[i] * (q->_.x[i] + q->_.x[i]);
		}
	}
	return 0;
}

int kad_op_sigm(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0].p;
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		p->n_d = q->n_d;
		memcpy(p->d, q->d, p->n_d * sizeof(int));
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i)
			p->_.x[i] = 1.0f / (1.0f + expf(q->_.x[i]));
	} else if (action == KAD_BACKWARD) {
		if (q->to_back) {
			float s = 1.0f / n;
			for (i = 0; i < n; ++i)
				q->g[i] += s * p->g[i] * (p->_.x[i] * (1.0f - p->_.x[i]));
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
		p->n_d = q->n_d;
		memcpy(p->d, q->d, p->n_d * sizeof(int));
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i) {
			float y;
			y = expf(-2.0f * q->_.x[i]);
			p->_.x[i] = (1.0f - y) / (1.0f + y);
		}
	} else if (action == KAD_BACKWARD) {
		if (q->to_back)
			for (i = 0; i < n; ++i)
				q->g[i] += p->g[i] * (1.0f - p->_.x[i] * p->_.x[i]);
	}
	return 0;
}

int kad_op_relu(kad_node_t *p, int action)
{
	int i, n;
	kad_node_t *q = p->child[0].p;
	n = kad_len(q);
	if (action == KAD_SYNC_DIM) {
		p->n_d = q->n_d;
		memcpy(p->d, q->d, p->n_d * sizeof(int));
	} else if (action == KAD_FORWARD) {
		for (i = 0; i < n; ++i)
			p->_.x[i] = q->_.x[i] > 0.0f? q->_.x[i] : 0.0f;
	} else if (action == KAD_BACKWARD) {
		if (q->to_back)
			for (i = 0; i < n; ++i)
				if (q->_.x[i] > 0.0f)
					q->g[i] += p->g[i];
	}
	return 0;
}

kad_op_f kad_op_list[] = {
	0,
	kad_op_add,
	kad_op_mul,
	kad_op_cmul,
	kad_op_ce2,
	kad_op_norm2,
	kad_op_sigm,
	kad_op_tanh,
	kad_op_relu,
	0
};
