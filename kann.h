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

#ifndef KANN_H
#define KANN_H

#define KANN_VERSION "r172"

#define KANN_L_IN       1   // input
#define KANN_L_OUT      2   // output
#define KANN_L_TRUTH    3   // truth output
#define KANN_L_COST     4   // final cost

#define KANN_H_TEMP     11  // temperature for softmax
#define KANN_H_DROPOUT  12  // dropout ratio
#define KANN_H_L2REG    13  // coefficient for L2 regulation

#define KANN_C_CEB      1   // binary cross-entropy cost, used with sigmoid
#define KANN_C_CEM      2   // multi-class cross-entropy cost, used with softmax

#define KANN_RDR_BATCH_RESET     1
#define KANN_RDR_MINI_RESET      2
#define KANN_RDR_READ_TRAIN      3
#define KANN_RDR_READ_VALIDATE   4

#define KANN_MM_DEFAULT 0
#define KANN_MM_SGD     1
#define KANN_MM_RMSPROP 2

#define KANN_MB_DEFAULT 0
#define KANN_MB_CONST   1

#include "kautodiff.h"

typedef struct {
	int n;
	kad_node_t **v;
	float *t, *g, *c;
} kann_t;

typedef struct {
	float lr;    // learning rate
	float fv;    // fraction of validation data
	int max_mbs; // max mini-batch size
	int max_rnn_len;
	int epoch_lazy;
	int max_epoch;

	float decay;
} kann_mopt_t;

typedef int (*kann_reader_f)(void *data, int action, int max_len, float *x, float *y);

#define kann_n_par(a) (kad_n_var((a)->n, (a)->v))
#define kann_is_hyper(p) ((p)->label == KANN_H_TEMP || (p)->label == KANN_H_DROPOUT || (p)->label == KANN_H_L2REG)

extern int kann_verbose;

#ifdef __cplusplus
extern "C" {
#endif

// common layers
kad_node_t *kann_layer_input(int n1);
kad_node_t *kann_layer_linear(kad_node_t *in, int n1);
kad_node_t *kann_layer_dropout(kad_node_t *t, float r);
kad_node_t *kann_layer_rnn(kad_node_t *in, int n1, kad_node_t *(*af)(kad_node_t*));
kad_node_t *kann_layer_gru(kad_node_t *in, int n1);
kann_t *kann_layer_final(kad_node_t *t, int n_out, int cost_type);

// basic model allocation/deallocation
void kann_set_hyper(kann_t *a, int label, float z);
void kann_delete(kann_t *a);

// number of input and output variables
int kann_n_in(const kann_t *a);
int kann_n_out(const kann_t *a);
int kann_n_hyper(const kann_t *a);

// unroll an RNN to an FNN
kann_t *kann_rnn_unroll(kann_t *a, int len);
void kann_delete_unrolled(kann_t *a);

// train a model
void kann_mopt_init(kann_mopt_t *mo);
void kann_train(const kann_mopt_t *mo, kann_t *a, kann_reader_f rdr, void *data);
void kann_fnn_train(const kann_mopt_t *mo, kann_t *a, int n, float **x, float **y);

// apply a trained model
const float *kann_apply1(kann_t *a, float *x);
void kann_rnn_start(kann_t *a);
void kann_rnn_end(kann_t *a);
float *kann_rnn_apply_seq1(kann_t *a, int len, float *x);

// model I/O
void kann_write_core(FILE *fp, kann_t *ann);
void kann_write(const char *fn, kann_t *ann);
kann_t *kann_read_core(FILE *fp);
kann_t *kann_read(const char *fn);

// pseudo-random number generator
void kann_srand(uint64_t seed);
uint64_t kann_rand(void);
double kann_drand(void);
double kann_normal(void);

#ifdef __cplusplus
}
#endif

#endif
