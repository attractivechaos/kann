#include <stdlib.h>
#include <math.h>
#include "autodiff.h"

struct ad_tnode_t {
	int n_row, n_col;
	int n_child, op;
	const float *x;
	struct ad_tnode_t *parent, **child;
};

static inline ad_tnode_t *ad_new_core(int n_row, int n_col, int n_child, const float *x)
{
	ad_tnode_t *s;
	s = (ad_tnode_t*)calloc(1, sizeof(ad_tnode_t));
	s->n_row = n_row, s->n_col = n_col, s->x = x;
	s->n_child = n_child;
	if (s->n_child) s->child = (ad_tnode_t**)calloc(s->n_child, sizeof(ad_tnode_t*));
	return s;
}

ad_tnode_t *ad_new(int n_row, int n_col, const float *x)
{
	return ad_new_core(n_row, n_col, 0, x);
}

void ad_del(ad_tnode_t *s)
{
	free(s->child);
	free(s);
}

static inline ad_tnode_t *ad_op1_core(int op, int n_row, int n_col, ad_tnode_t *x)
{
	ad_tnode_t *s;
	s = ad_new_core(n_row, n_col, 1, 0);
	s->op = op;
	s->child[0] = x;
	x->parent = s;
	return s;
}

static inline ad_tnode_t *ad_op2_core(int op, int n_row, int n_col, ad_tnode_t *x, ad_tnode_t *y)
{
	ad_tnode_t *s;
	s = ad_new_core(n_row, n_col, 2, 0);
	s->op = op;
	s->child[0] = x, s->child[1] = y;
	x->parent = y->parent = s;
	return s;
}

ad_tnode_t *ad_add(ad_tnode_t *x, ad_tnode_t *y)
{
	return (x->n_row != y->n_row || x->n_col != y->n_col)? 0 : ad_op2_core(AD_OP_ADD, x->n_row, x->n_col, x, y);
}

ad_tnode_t *ad_sub(ad_tnode_t *x, ad_tnode_t *y)
{
	return (x->n_row != y->n_row || x->n_col != y->n_col)? 0 : ad_op2_core(AD_OP_SUB, x->n_row, x->n_col, x, y);
}

ad_tnode_t *ad_hmul(ad_tnode_t *x, ad_tnode_t *y)
{
	return (x->n_row != y->n_row || x->n_col != y->n_col)? 0 : ad_op2_core(AD_OP_HMUL, x->n_row, x->n_col, x, y);
}

ad_tnode_t *ad_mul(ad_tnode_t *x, ad_tnode_t *y)
{
	return (x->n_col != y->n_row)? 0 : ad_op2_core(AD_OP_MUL, x->n_row, y->n_col, x, y);
}

ad_tnode_t *ad_smul(ad_tnode_t *x, ad_tnode_t *y)
{
	if (x->n_row == 1 && x->n_col == 1) return ad_op2_core(AD_OP_SMUL, y->n_row, y->n_col, x, y);
	if (y->n_row == 1 && y->n_col == 1) return ad_op2_core(AD_OP_SMUL, x->n_row, x->n_col, y, x);
	return 0;
}

ad_tnode_t *ad_dot(ad_tnode_t *x, ad_tnode_t *y)
{
	return (x->n_row != y->n_row || x->n_col != y->n_col)? 0 : ad_op2_core(AD_OP_DOT, 1, x->n_col, x, y);
}

ad_tnode_t *ad_square(ad_tnode_t *x)
{
	return ad_op1_core(AD_OP_SQUARE, 1, x->n_col, x);
}

ad_tnode_t *ad_sigm(ad_tnode_t *x)
{
	return ad_op1_core(AD_OP_SIGM, x->n_row, x->n_col, x);
}
