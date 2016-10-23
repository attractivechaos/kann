#ifndef KANN_AUTODIFF_H
#define KANN_AUTODIFF_H

#include <stdint.h>

struct ad_node_t;
typedef struct ad_node_t ad_node_t;

struct ad_expr_t;
typedef struct ad_expr_t ad_expr_t;

#ifdef __cplusplus
extern "C" {
#endif

ad_node_t *ad_var(int n_row, int n_col, const float *x);
ad_node_t *ad_param(int n_row, int n_col, const float *x);

ad_node_t *ad_add(ad_node_t *x, ad_node_t *y);
ad_node_t *ad_sub(ad_node_t *x, ad_node_t *y);
ad_node_t *ad_hmul(ad_node_t *x, ad_node_t *y);
ad_node_t *ad_mul(ad_node_t *x, ad_node_t *y);
ad_node_t *ad_smul(ad_node_t *x, ad_node_t *y);
ad_node_t *ad_dot(ad_node_t *x, ad_node_t *y);

ad_node_t *ad_square(ad_node_t *x);
ad_node_t *ad_sigm(ad_node_t *x);

ad_expr_t *ad_expr_compile(ad_node_t *root);
void ad_expr_destroy(ad_expr_t *e);
void ad_expr_debug(const ad_expr_t *e);
const float *ad_expr_forward(ad_expr_t *e);
void ad_expr_backward(ad_expr_t *e);

#ifdef __cplusplus
}
#endif

#endif
