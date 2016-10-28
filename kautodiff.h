#ifndef KANN_AUTODIFF_H
#define KANN_AUTODIFF_H

#include <stdio.h>

#define KAD_ALLOC      1
#define KAD_FORWARD    2
#define KAD_BACKWARD   3
#define KAD_SYNC_SHAPE 4

struct kad_node_t;

typedef struct {
	struct kad_node_t *p;
	float *t;
} kad_edge_t;

typedef struct kad_node_t {
	int n_row, n_col, op;
	int label, tmp;
	short n_child, to_back;
	union {
		const float *cx;
		float *x;
	} _;
	float *d;
	kad_edge_t *child;
} kad_node_t;

typedef int (*kad_op_f)(kad_node_t*, int);
extern kad_op_f kad_op_list[];

#ifdef __cplusplus
extern "C" {
#endif

kad_node_t *kad_par(int n_row, int n_col, const float *x);
kad_node_t *kad_var(int n_row, int n_col, const float *x, float *d);

kad_node_t *kad_add(kad_node_t *x, kad_node_t *y);   // z(x,y) = x + y (element-wise/matrix addition)
kad_node_t *kad_sub(kad_node_t *x, kad_node_t *y);   // z(x,y) = x - y (element-wise/matrix subtraction)
kad_node_t *kad_mul(kad_node_t *x, kad_node_t *y);   // z(x,y) = x * y (element-wise product)
kad_node_t *kad_mtmul(kad_node_t *x, kad_node_t *y); // z(x,y) = x * y^T (general matrix product, with y transposed; only y is differentiable)
kad_node_t *kad_ce2(kad_node_t *x, kad_node_t *y);   // z(x,y) = \sum_i -y_i*log(f(x_i)) - (1-y_i)*log(1-f(x_i)); f() is sigmoid (binary cross-entropy for sigmoid; only x differentiable)

kad_node_t *kad_norm2(kad_node_t *x);               // z(x) = \sum_i x_i^2 (L2 norm)
kad_node_t *kad_sigm(kad_node_t *x);                // z(x) = 1/(1+exp(-x)) (element-wise sigmoid)
kad_node_t *kad_tanh(kad_node_t *x);                // z(x) = (1-exp(-2x)) / (1+exp(-2x)) (element-wise tanh)
kad_node_t *kad_relu(kad_node_t *x);                // z(x) = max{0,x} (element-wise rectifier (aka ReLU))

kad_node_t **kad_compile(kad_node_t *root, int *n_node);
float kad_eval(int n, kad_node_t **a, int cal_grad);
void kad_free(int n, kad_node_t **a);

int kad_write(FILE *fp, int n_node, kad_node_t **node);
kad_node_t **kad_read(FILE *fp, int *_n_node);

#ifdef __cplusplus
}
#endif

#endif
