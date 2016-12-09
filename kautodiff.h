/*
  The MIT License

  Copyright (c) 2016  Broad Institute

  Permission is hereby granted, free of charge, to any person obtaining
  a copy of this software and associated documentation files (the
  "Software"), to deal in the Software without restriction, including
  without limitation the rights to use, copy, modify, merge, publish,
  distribute, sublicense, and/or sell copies of the Software, and to
  permit persons to whom the Software is furnished to do so, subject to
  the following conditions:

  The above copyright notice and this permission notice shall be
  included in all copies or substantial portions of the Software.

  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
  ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
  CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

#ifndef KANN_AUTODIFF_H
#define KANN_AUTODIFF_H

#define KAD_VERSION "r223"

#include <stdio.h>
#include <stdint.h>

#define KAD_MAX_DIM 4     // max dimension
#define KAD_MAX_OP  64    // max number of operators

struct kad_node_t;
typedef struct kad_node_t kad_node_t;

/* A computational graph is an acyclic directed graph. In the graph, an
 * external node represents a differentiable variable or a non-differentiable
 * parameter; an internal node represents an operator; an edge from node v to w
 * indicates v is an operand of w.
 */

// an edge between two nodes in the computational graph
typedef struct {
	kad_node_t *p;  // child node, not allocated
	float *t;       // temporary data needed for backprop; allocated on heap if not NULL
} kad_edge_t;

// a node in the computational graph
struct kad_node_t {
	uint8_t     n_d;            // number of dimensions; no larger than KAD_MAX_DIM
	uint8_t     to_back;        // whether to do back propogation (boolean)
	uint16_t    op;             // operator; kad_op_list[op] is the actual function
	uint32_t    n_child;        // number of operands/child nodes
	int32_t     label;          // label for external uses
	int32_t     tmp;            // temporary field; MUST BE zero before calling kad_compile()
	int32_t     d[KAD_MAX_DIM]; // dimensions
	float      *x;              // value; allocated for internal nodes
	float      *g;              // gradient; allocated for internal nodes
	kad_edge_t *child;          // operands/child nodes
	kad_node_t *pre;            // usually NULL; only used for RNN
	void       *ptr;            // for special operators that need additional parameters (e.g. conv2d)
};

#define KAD_ALLOC      1
#define KAD_FORWARD    2
#define KAD_BACKWARD   3
#define KAD_SYNC_DIM   4

typedef int (*kad_op_f)(kad_node_t*, int);
extern kad_op_f kad_op_list[KAD_MAX_OP];

typedef double (*kad_drand_f)(void);
extern kad_drand_f kad_drand; // random number generator, default to drand48()

#define kad_is_var(p) ((p)->n_child == 0 && (p)->to_back)
#define kad_is_pool(p) ((p)->op == 10)

#ifdef __cplusplus
extern "C" {
#endif

// define a variable (differentiable) or a parameter (not differentiable)
kad_node_t *kad_par(float *x, int n_d, ...);
kad_node_t *kad_var(float *x, float *g, int n_d, ...);

// operators taking two operands
kad_node_t *kad_add(kad_node_t *x, kad_node_t *y);   // f(x,y) = x + y       (element-wise addition)
kad_node_t *kad_mul(kad_node_t *x, kad_node_t *y);   // f(x,y) = x * y       (element-wise product)
kad_node_t *kad_cmul(kad_node_t *x, kad_node_t *y);  // f(x,y) = x * y^T     (column-wise matrix product; i.e. y is transposed)
kad_node_t *kad_ceb(kad_node_t *x, kad_node_t *y);   // f(x,y) = \sum_i -y_i*log(s(x_i)) - (1-y_i)*log(1-s(x_i))  (s() is sigmoid; binary cross-entropy for sigmoid; only x differentiable)
kad_node_t *kad_cem(kad_node_t *x, kad_node_t *y);   // f(x,y) = - \sum_i -y_i*log(s(x_i))  (s() is softmax; cross-entropy for softmax; only x differentiable)
kad_node_t *kad_softmax2(kad_node_t *x, kad_node_t *y); // softmax with temperature
kad_node_t *kad_dropout(kad_node_t *x, kad_node_t *r);  // dropout at rate r
kad_node_t *kad_conv2d(kad_node_t *x, kad_node_t *w, int stride, int pad);
kad_node_t *kad_max2d(kad_node_t *x, kad_node_t *m, int stride, int pad);

// operators taking one operand
kad_node_t *kad_norm2(kad_node_t *x);  // f(x) = \sum_i x_i^2                (L2 norm)
kad_node_t *kad_sigm(kad_node_t *x);   // f(x) = 1/(1+exp(-x))               (element-wise sigmoid)
kad_node_t *kad_tanh(kad_node_t *x);   // f(x) = (1-exp(-2x)) / (1+exp(-2x)) (element-wise tanh)
kad_node_t *kad_relu(kad_node_t *x);   // f(x) = max{0,x}                    (element-wise rectifier, aka ReLU)
kad_node_t *kad_1minus(kad_node_t *x); // f(x) = 1 - x
kad_node_t *kad_softmax(kad_node_t *x);// softmax without temperature (i.e. temperature==1)

// operators taking an indefinite number of operands (mostly for pooling)
kad_node_t *kad_avg(int n, kad_node_t **x); // f(x_1,...,x_n) = \sum_i x_i/n (mean pooling)

// compile graph and graph deallocation
kad_node_t **kad_compile_array(int *n_node, int n_roots, kad_node_t **roots);
kad_node_t **kad_compile(int *n_node, int n_roots, ...);
void kad_delete(int n, kad_node_t **a);
void kad_allocate_internal(int n, kad_node_t **v);

// operations on compiled graph
const float *kad_eval_from(int n, kad_node_t **a, int from);
void kad_eval_by_label(int n, kad_node_t **a, int label);
void kad_grad(int n, kad_node_t **a, int from);

// autodiff graph I/O
int kad_write(FILE *fp, int n_node, kad_node_t **node);
kad_node_t **kad_read(FILE *fp, int *_n_node);

// debugging routines
void kad_trap_fe(void); // abort on divide-by-zero and NaN
void kad_print_graph(FILE *fp, int n, kad_node_t **v);
void kad_check_grad(int n, kad_node_t **a, int from);

#ifdef __cplusplus
}
#endif

static inline int kad_len(const kad_node_t *p) // calculate the size of p->x
{
	int n = 1, i;
	for (i = 0; i < p->n_d; ++i) n *= p->d[i];
	return n;
}

static inline int kad_n_var(int n, kad_node_t *const* v) // total number of variables in the graph
{
	int c = 0, i;
	for (i = 0; i < n; ++i)
		if (v[i]->n_child == 0 && v[i]->to_back)
			c += kad_len(v[i]);
	return c;
}

#endif
