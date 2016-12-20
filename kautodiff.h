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

#define KAD_VERSION "r280"

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

#define KAD_F_WITH_PD  0x1 // PD = partial derivative
#define KAD_F_CONSTANT 0x2
#define KAD_F_POOLING  0x4

#define kad_is_back(p)  ((p)->flag & KAD_F_WITH_PD)
#define kad_is_ext(p)   ((p)->n_child == 0)
#define kad_is_var(p)   (kad_is_ext(p) && kad_is_back(p))
#define kad_is_const(p) (kad_is_ext(p) && ((p)->flag & KAD_F_CONSTANT))
#define kad_is_feed(p)  (kad_is_ext(p) && !kad_is_back(p) && !((p)->flag & KAD_F_CONSTANT))
#define kad_is_pivot(p) ((p)->n_child == 1 && ((p)->flag & KAD_F_POOLING))

// a node in the computational graph
struct kad_node_t {
	uint8_t     n_d;            // number of dimensions; no larger than KAD_MAX_DIM
	uint8_t     flag;           // type of the node; see KAD_F_* for valid flags
	uint16_t    op;             // operator; kad_op_list[op] is the actual function
	int32_t     n_child;        // number of operands/child nodes
	int32_t     tmp;            // temporary field; MUST BE zero before calling kad_compile()
	int32_t     ptr_size;       // size of ptr below
	int32_t     d[KAD_MAX_DIM]; // dimensions
	int32_t     ext_label;
	uint32_t    ext_flag;
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

#define KAD_PAD_NONE  0
#define KAD_PAD_AUTO  (-1)
#define KAD_PAD_SAME  (-2)

#ifdef __cplusplus
extern "C" {
#endif

// define a variable, a constant or a feed (placeholder in TensorFlow)
kad_node_t *kad_var(float *x, float *g, int n_d, ...);
kad_node_t *kad_const(float *x, int n_d, ...);
kad_node_t *kad_feed(float *x, int n_d, ...);

// operators taking two operands
kad_node_t *kad_add(kad_node_t *x, kad_node_t *y);   // f(x,y) = x + y       (element-wise addition)
kad_node_t *kad_mul(kad_node_t *x, kad_node_t *y);   // f(x,y) = x * y       (element-wise product)
kad_node_t *kad_cmul(kad_node_t *x, kad_node_t *y);  // f(x,y) = x * y^T     (column-wise matrix product; i.e. y is transposed)
kad_node_t *kad_matmul(kad_node_t *x, kad_node_t *y);// f(x,y) = x * y
kad_node_t *kad_ceb(kad_node_t *x, kad_node_t *y);   // f(x,y) = \sum_i -y_i*log(s(x_i)) - (1-y_i)*log(1-s(x_i))  (s() is sigmoid; binary cross-entropy for sigmoid; only x differentiable)
kad_node_t *kad_cem(kad_node_t *x, kad_node_t *y);   // f(x,y) = - \sum_i -y_i*log(s(x_i))  (s() is softmax; cross-entropy for softmax; only x differentiable)
kad_node_t *kad_softmax2(kad_node_t *x, kad_node_t *y); // softmax with temperature
kad_node_t *kad_dropout(kad_node_t *x, kad_node_t *r);  // dropout at rate r
kad_node_t *kad_split(kad_node_t *x, int dim, int start, int end);
kad_node_t *kad_conv2d(kad_node_t *x, kad_node_t *w, int r_stride, int c_stride, int r_pad, int c_pad);
kad_node_t *kad_max2d(kad_node_t *x, int kernel_h, int kernel_w, int r_stride, int c_stride, int r_pad, int c_pad);
kad_node_t *kad_conv1d(kad_node_t *x, kad_node_t *w, int stride, int pad);
kad_node_t *kad_max1d(kad_node_t *x, int kernel_size, int stride, int pad);

// operators taking one operand
kad_node_t *kad_norm2(kad_node_t *x);  // f(x) = \sum_i x_i^2                (L2 norm)
kad_node_t *kad_sigm(kad_node_t *x);   // f(x) = 1/(1+exp(-x))               (element-wise sigmoid)
kad_node_t *kad_tanh(kad_node_t *x);   // f(x) = (1-exp(-2x)) / (1+exp(-2x)) (element-wise tanh)
kad_node_t *kad_relu(kad_node_t *x);   // f(x) = max{0,x}                    (element-wise rectifier, aka ReLU)
kad_node_t *kad_1minus(kad_node_t *x); // f(x) = 1 - x
kad_node_t *kad_softmax(kad_node_t *x);// softmax without temperature (i.e. temperature==1)

// operators taking an indefinite number of operands (mostly for pooling)
kad_node_t *kad_avg(int n, kad_node_t **x); // f(x_1,...,x_n) = \sum_i x_i/n (mean pooling)
kad_node_t *kad_max(int n, kad_node_t **x);

// compile graph
kad_node_t **kad_compile_array(int *n_node, int n_roots, kad_node_t **roots);
kad_node_t **kad_compile(int *n_node, int n_roots, ...);
void kad_delete(int n, kad_node_t **a);

// compute values and gradients
const float *kad_eval_at(int n, kad_node_t **a, int from);
void kad_eval_flag(int n, kad_node_t **a, int ext_flag);
void kad_grad(int n, kad_node_t **a, int from);

// miscellaneous operations on a compiled graph
int kad_n_var(int n, kad_node_t *const* v);
int kad_n_const(int n, kad_node_t *const* v);
void kad_ext_collate(int n, kad_node_t **v, float **_x, float **_g, float **_c);
void kad_ext_sync(int n, kad_node_t **v, float *x, float *g, float *c);
kad_node_t **kad_unroll(int n_v, kad_node_t **v, int len, int *new_n);

// graph I/O
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

#endif
