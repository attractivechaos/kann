#ifndef KANN_AUTODIFF_H
#define KANN_AUTODIFF_H

#define KAD_VERSION "r77"

#include <stdio.h>

#define KAD_MAX_DIM 4     // max dimension

/* An autodiff graph is a directed acyclic graph (DAG), where an external node
 * represents a variable (differentiable) or a parameter (not differentiable),
 * and an internal node represents an operator. Each node is associated with a
 * value, which is a single-precision N-dimensional array. An operator is the
 * parent of all its operands. A node without parents is a terminal node. The
 * graph may have one or multiple terminal nodes.
 */

struct kad_node_t;
typedef struct kad_node_t kad_node_t;

// an edge between two nodes in the autodiff graph
typedef struct {
	kad_node_t *p;        // child node
	float *t;             // temporary data needed for backprop; allocated if not NULL
} kad_edge_t;

// a node in the autodiff graph
struct kad_node_t {
	short n_d;            // number of dimensions; no larger than KAD_MAX_DIM
	short op;             // operator; kad_op_list[op] is the actual function
	int label;            // label for external uses
	int tmp;              // temporary field; MUST BE zero before calling kad_compile()
	short n_child;        // number of child nodes
	short to_back;        // whether to do back propogation
	int d[KAD_MAX_DIM];   // dimensions
	float *x;             // value; allocated for internal nodes
	float *g;             // gradient; allocated for internal nodes
	kad_edge_t *child;    // child nodes
	kad_node_t *pre;      // usually NULL; only used when unrolling an RNN
	void *ptr;            // auxiliary data
};

#define KAD_ALLOC      1
#define KAD_FORWARD    2
#define KAD_BACKWARD   3
#define KAD_SYNC_DIM   4

typedef int (*kad_op_f)(kad_node_t*, int);
extern kad_op_f kad_op_list[];

#define kad_is_var(p) ((p)->n_child == 0 && (p)->to_back)

typedef struct {
	void *data;
	double (*func)(void*);
} kad_rng_t;

#ifdef __cplusplus
extern "C" {
#endif

kad_node_t *kad_par(float *x, int n_d, ...);
kad_node_t *kad_var(float *x, float *g, int n_d, ...);

kad_node_t *kad_add(kad_node_t *x, kad_node_t *y);   // z(x,y) = x + y (element-wise addition)
kad_node_t *kad_mul(kad_node_t *x, kad_node_t *y);   // z(x,y) = x * y (element-wise product)
kad_node_t *kad_cmul(kad_node_t *x, kad_node_t *y);  // z(x,y) = x * y^T (matrix product, with y transposed)
kad_node_t *kad_ce2(kad_node_t *x, kad_node_t *y);   // z(x,y) = \sum_i -y_i*log(f(x_i)) - (1-y_i)*log(1-f(x_i)); f() is sigmoid (binary cross-entropy for sigmoid; only x differentiable)

kad_node_t *kad_norm2(kad_node_t *x); // z(x) = \sum_i x_i^2 (L2 norm)
kad_node_t *kad_sigm(kad_node_t *x);  // z(x) = 1/(1+exp(-x)) (element-wise sigmoid)
kad_node_t *kad_tanh(kad_node_t *x);  // z(x) = (1-exp(-2x)) / (1+exp(-2x)) (element-wise tanh)
kad_node_t *kad_relu(kad_node_t *x);  // z(x) = max{0,x} (element-wise rectifier (aka ReLU))

kad_node_t **kad_compile(int *n_node, int n_roots, ...);
kad_node_t **kad_unroll(int n, kad_node_t **v, int len, int *new_n);
const float *kad_eval(int n, kad_node_t **a, int from);
void kad_grad(int n, kad_node_t **a, int from);
void kad_free_node(kad_node_t *p);
void kad_free(int n, kad_node_t **a);

// autodiff graph I/O
void kad_write1(FILE *fp, const kad_node_t *p);
kad_node_t *kad_read1(FILE *fp, kad_node_t **node);
int kad_write(FILE *fp, int n_node, kad_node_t **node);
kad_node_t **kad_read(FILE *fp, int *_n_node);

// vector operations
float kad_sdot(int n, const float *x, const float *y);
void kad_saxpy(int n, float a, const float *x, float *y);

// defined in kad_debug.c
void kad_debug(FILE *fp, int n, kad_node_t **v);
void kad_check_grad(int n, kad_node_t **a, int from);

#ifdef __cplusplus
}
#endif

static inline int kad_len(const kad_node_t *p)
{
	int n = 1, i;
	for (i = 0; i < p->n_d; ++i) n *= p->d[i];
	return n;
}

static inline int kad_n_var(int n, kad_node_t *const* v)
{
	int c = 0, i;
	for (i = 0; i < n; ++i)
		if (v[i]->n_child == 0 && v[i]->to_back)
			c += kad_len(v[i]);
	return c;
}

#endif
