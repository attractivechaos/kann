#ifndef KANN_AUTODIFF_H
#define KANN_AUTODIFF_H

#define AD_OP_ADD      1
#define AD_OP_SUB      2
#define AD_OP_MUL      3   // matrix product
#define AD_OP_SMUL     4   // scalar-matrix product
#define AD_OP_HMUL     5   // Hadamard product
#define AD_OP_DOT      6   // vector dot (inner product)
#define AD_OP_SQUARE   101
#define AD_OP_SIGM     102

struct ad_tnode_t;
typedef struct ad_tnode_t ad_tnode_t;

#ifdef __cplusplus
extern "C" {
#endif

ad_tnode_t *ad_new(int n_row, int n_col, const float *x);
ad_tnode_t *ad_add(ad_tnode_t *x, ad_tnode_t *y);
ad_tnode_t *ad_sub(ad_tnode_t *x, ad_tnode_t *y);
ad_tnode_t *ad_hmul(ad_tnode_t *x, ad_tnode_t *y);
ad_tnode_t *ad_mul(ad_tnode_t *x, ad_tnode_t *y);
ad_tnode_t *ad_smul(ad_tnode_t *x, ad_tnode_t *y);
ad_tnode_t *ad_dot(ad_tnode_t *x, ad_tnode_t *y);
ad_tnode_t *ad_square(ad_tnode_t *x);
ad_tnode_t *ad_sigm(ad_tnode_t *x);

#ifdef __cplusplus
}
#endif

#endif
