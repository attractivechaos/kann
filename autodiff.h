#ifndef KANN_AUTODIFF_H
#define KANN_AUTODIFF_H

#include <stdint.h>

#define AD_ALLOC    1
#define AD_FORWARD  2
#define AD_BACKWARD 3
#define AD_SYNCDIM  4

struct ad_node_t;

typedef struct {
	struct ad_node_t *p;
	float *t;
} ad_edge_t;

typedef struct ad_node_t {
	int n_row, n_col, op;
	int n_child, to_back;
	int cnt; // used for topological sorting
	union {
		const float *cx;
		float *x;
	} _;
	float *d;
	ad_edge_t *child;
} ad_node_t;

typedef int (*ad_op_f)(ad_node_t*, int);
extern ad_op_f ad_op_list[];

#ifdef __cplusplus
extern "C" {
#endif

ad_node_t *ad_par(int n_row, int n_col, const float *x);
ad_node_t *ad_var(int n_row, int n_col, const float *x, float *d);

ad_node_t *ad_add(ad_node_t *x, ad_node_t *y);   // z(x,y) = x + y (element-wise/matrix addition)
ad_node_t *ad_sub(ad_node_t *x, ad_node_t *y);   // z(x,y) = x - y (element-wise/matrix subtraction)
ad_node_t *ad_mul(ad_node_t *x, ad_node_t *y);   // z(x,y) = x * y (element-wise product)
ad_node_t *ad_mtmul(ad_node_t *x, ad_node_t *y); // z(x,y) = x * y^T (general matrix product, with y transposed)
ad_node_t *ad_smul(ad_node_t *x, ad_node_t *y);  // z(x,y) = x * y; x is a scalar (scalar-to-matrix product)
ad_node_t *ad_ce2(ad_node_t *x, ad_node_t *y);   // z(x,y) = \sum_i -y_i*log(f(x_i)) - (1-y_i)*log(1-f(x_i)); f() is sigmoid (binary cross-entropy for sigmoid)

ad_node_t *ad_norm2(ad_node_t *x);               // z(x) = \sum_i x_i^2 (L2 norm)
ad_node_t *ad_sigm(ad_node_t *x);                // z(x) = 1/(1+exp(-x)) (element-wise sigmoid)
ad_node_t *ad_tanh(ad_node_t *x);                // z(x) = (1-exp(-2x)) / (1+exp(-2x)) (element-wise tanh)

ad_node_t **ad_compile(ad_node_t *root, int *n_node);
float ad_eval(int n, ad_node_t **a);
void ad_free(int n, ad_node_t **a);

#ifdef __cplusplus
}
#endif

#endif
