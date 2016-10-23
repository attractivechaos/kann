#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "autodiff.h"

#define AD_OP_PARAM    -2
#define AD_OP_VAR      -1
#define AD_OP_NULL     0
#define AD_OP_ADD      1
#define AD_OP_SUB      2
#define AD_OP_MUL      3   // matrix product
#define AD_OP_SMUL     4   // scalar-matrix product
#define AD_OP_HMUL     5   // Hadamard product
#define AD_OP_DOT      6   // vector dot (inner product)
#define AD_OP_SQUARE   101
#define AD_OP_SIGM     102

/***************************
 * Expression construction *
 ***************************/

struct ad_node_t {
	int n_row, n_col;
	int n_child, op;
	const float *x;
	struct ad_node_t **child;
};

static inline ad_node_t *ad_new_core(int op, int n_row, int n_col, int n_child, const float *x)
{
	ad_node_t *s;
	s = (ad_node_t*)calloc(1, sizeof(ad_node_t));
	s->op = op, s->n_row = n_row, s->n_col = n_col, s->n_child = n_child, s->x = x;
	if (s->n_child) s->child = (ad_node_t**)calloc(s->n_child, sizeof(ad_node_t*));
	return s;
}

ad_node_t *ad_var(int n_row, int n_col, const float *x)
{
	return ad_new_core(AD_OP_VAR, n_row, n_col, 0, x);
}

ad_node_t *ad_param(int n_row, int n_col, const float *x)
{
	return ad_new_core(AD_OP_PARAM, n_row, n_col, 0, x);
}

static inline ad_node_t *ad_op2_core(int op, int n_row, int n_col, ad_node_t *x, ad_node_t *y)
{
	ad_node_t *s;
	s = ad_new_core(op, n_row, n_col, 2, 0);
	s->child[0] = x, s->child[1] = y;
	return s;
}

#define AD_FUNC_OP2(fname, op, cond, _row, _col) \
	ad_node_t *fname(ad_node_t *x, ad_node_t *y) { return (cond)? 0 : ad_op2_core((op), (_row), (_col), x, y); }

AD_FUNC_OP2(ad_add, AD_OP_ADD, (x->n_row != y->n_row || x->n_col != y->n_col), x->n_row, x->n_col)
AD_FUNC_OP2(ad_sub, AD_OP_SUB, (x->n_row != y->n_row || x->n_col != y->n_col), x->n_row, x->n_col)
AD_FUNC_OP2(ad_hmul, AD_OP_HMUL, (x->n_row != y->n_row || x->n_col != y->n_col), x->n_row, x->n_col)
AD_FUNC_OP2(ad_mul, AD_OP_MUL, (x->n_col != y->n_row), x->n_row, y->n_col)
AD_FUNC_OP2(ad_smul, AD_OP_SMUL, (x->n_row == 1 && x->n_col == 1), y->n_row, y->n_col)
AD_FUNC_OP2(ad_dot, AD_OP_DOT, (x->n_row != y->n_row || x->n_col != y->n_col), 1, x->n_col)

static inline ad_node_t *ad_op1_core(int op, int n_row, int n_col, ad_node_t *x)
{
	ad_node_t *s;
	s = ad_new_core(op, n_row, n_col, 1, 0);
	s->child[0] = x;
	return s;
}

#define AD_FUNC_OP1(fname, op, _row, _col) \
	ad_node_t *fname(ad_node_t *x) { return ad_op1_core((op), (_row), (_col), x); }

AD_FUNC_OP1(ad_square, AD_OP_SQUARE, 1, x->n_col)
AD_FUNC_OP1(ad_sigm, AD_OP_SIGM, x->n_row, x->n_col)

/**************************
 * Expression compilation *
 **************************/

#define kvec_t(type) struct { size_t n, m; type *a; }

#define kv_pop(v) ((v).a[--(v).n])
#define kv_pushp(type, v, p) do { \
		if ((v).n == (v).m) { \
			(v).m = (v).m? (v).m<<1 : 2; \
			(v).a = (type*)realloc((v).a, sizeof(type) * (v).m); \
		} \
		*(p) = &(v).a[(v).n++]; \
	} while (0)

typedef struct {
	int op, n_operand, n_row, n_col;
	union {
		const float *cx;
		float *x;
	} _;
} ad_expr1_t;

struct ad_expr_t {
	int32_t n;
	ad_expr1_t *a;
};

typedef struct {
	ad_node_t *p;
	int n;
} ad_tnode_t;

ad_expr_t *ad_expr_compile(ad_node_t *root)
{
	kvec_t(ad_tnode_t) stack = {0,0,0};
	kvec_t(ad_expr1_t) expr = {0,0,0};
	ad_tnode_t *p;
	ad_expr1_t *q;
	ad_expr_t *e;

	kv_pushp(ad_tnode_t, stack, &p);
	p->p = root, p->n = 0;
	while (stack.n) {
		ad_tnode_t *t = &stack.a[stack.n-1];
		if (t->n == t->p->n_child) {
			kv_pushp(ad_expr1_t, expr, &q);
			q->op = t->p->op;
			q->n_operand = t->p->n_child;
			q->n_row = t->p->n_row, q->n_col = t->p->n_col;
			q->_.cx = t->p->x;
			free(t->p->child);
			free(t->p);
			--stack.n;
		} else {
			kv_pushp(ad_tnode_t, stack, &p);
			p->p = t->p->child[t->n], p->n = 0;
			++t->n;
		}
	}
	free(stack.a);
	e = (ad_expr_t*)calloc(1, sizeof(ad_expr_t));
	e->n = expr.n;
	e->a = expr.a;
	return e;
}

void ad_expr_destroy(ad_expr_t *e)
{
	int32_t i;
	for (i = 0; i < e->n; ++i) {
		if (e->a[i].op > 0)
			free(e->a[i]._.x);
	}
	free(e->a);
	free(e);
}

void ad_expr_debug(const ad_expr_t *e)
{
	int32_t i;
	for (i = 0; i < e->n; ++i) {
		if (i > 0) printf(" ");
		printf("%d,%d", e->a[i].op, e->a[i].n_operand);
	}
	putchar('\n');
}
