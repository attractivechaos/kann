#ifndef KANN_AUTODIFF_H
#define KANN_AUTODIFF_H

#include <stdint.h>

#define AD_DT_IDEN    1  // identity matrix
#define AD_DT_NEGIDEN 2  // negative identity matrix
#define AD_DT_DIAG    3  // diagonal matrix
#define AD_DT_VEC     4
#define AD_DT_OUTMAT  5  // I x A^T
#define AD_DT_MATOUT  6  // A x I

struct ad_node_t;

typedef struct {
	int dtype;
	struct ad_node_t *p;
	float *z;
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

#ifdef __cplusplus
extern "C" {
#endif

ad_node_t *ad_var(int n_row, int n_col, const float *x, float *d);
ad_node_t *ad_param(int n_row, int n_col, const float *x);

ad_node_t *ad_add(ad_node_t *x, ad_node_t *y);
ad_node_t *ad_sub(ad_node_t *x, ad_node_t *y);
ad_node_t *ad_mul(ad_node_t *x, ad_node_t *y);
ad_node_t *ad_mmul(ad_node_t *x, ad_node_t *y);

ad_node_t *ad_norm2(ad_node_t *x);
ad_node_t *ad_sigm(ad_node_t *x);

ad_node_t **ad_compile(ad_node_t *root, int *n_node);
float ad_eval(int n, ad_node_t **a);
void ad_free(int n, ad_node_t **a);

#ifdef __cplusplus
}
#endif

#endif
